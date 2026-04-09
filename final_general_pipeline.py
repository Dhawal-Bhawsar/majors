"""
Final generalized Parkinson pipeline
===================================

What this script does:
- Builds numerical features (281 per segment)
- Builds graphical features (spectrogram, mel-spectrogram, MFCC, STFT)
- Uses a proper file-level 80/20 train/test split
- Trains ML models without cross-validation
- Trains 3 generalized CNN models on the same merged graphical dataset
    (VGG19, ResNet50, MobileNet)
- Saves models and preprocessing artifacts for reuse on new datasets

Outputs are written to final_pipeline_outputs/.
"""

import json
import os
import shutil
import warnings
from datetime import datetime
from collections import defaultdict

import joblib
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.applications import MobileNet, ResNet50, VGG19
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "26_29_09_2017_KCL", "26-29_09_2017_KCL")
CATEGORIES = ["ReadText", "SpontaneousDialogue"]
CLASSES = {"HC": 0, "PD": 1}
CLASS_NAMES = {0: "HC", 1: "PD"}

RANDOM_STATE = 42
SR = 22050
IMG_SIZE = (100, 100)
BATCH_SIZE = 16
EPOCHS = 35
N_FOLDS = 5

GRAPHICAL_CNN_MODELS = ["VGG19", "ResNet50", "MobileNet"]

# Segments to evaluate for the numerical pipeline.
# You can trim this list if you want a faster run.
SEGMENT_DURATIONS = [5, 10, 15, 30, 60]

# Final output folders
OUTPUT_DIR = os.path.join(BASE_DIR, "final_pipeline_outputs")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
GRAPHICS_DIR = os.path.join(OUTPUT_DIR, "graphics")
GRAPHICS_COMBINED_DIR = os.path.join(OUTPUT_DIR, "graphics_combined")
MODELS_DIR = os.path.join(OUTPUT_DIR, "cnn_models")
ML_MODELS_DIR = os.path.join(OUTPUT_DIR, "ml_models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
for d in [OUTPUT_DIR, FEATURES_DIR, GRAPHICS_DIR, GRAPHICS_COMBINED_DIR, MODELS_DIR, ML_MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# =============================================================================
# UTILITIES
# =============================================================================
def print_section(title):
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def print_subsection(title):
    print(f"\n--- {title} ---")


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def clean_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_audio(file_path, sr=SR):
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def segment_audio(y, duration_sec):
    samples = duration_sec * SR
    if len(y) < samples:
        return [y]
    n_segs = len(y) // samples
    return [y[i * samples:(i + 1) * samples] for i in range(n_segs)]


def collect_files():
    files = []
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: {DATASET_DIR} not found")
        return files

    for cat in CATEGORIES:
        cat_path = os.path.join(DATASET_DIR, cat)
        if not os.path.exists(cat_path):
            continue
        for cls, lbl in CLASSES.items():
            folder = os.path.join(cat_path, cls)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(".wav"):
                    files.append((os.path.join(folder, fname), lbl, cat))

    return files


def split_files(files, test_size=0.2):
    labels = [lbl for _, lbl, _ in files]
    train_files, test_files = train_test_split(
        files,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return train_files, test_files


def split_train_val(files, val_size=0.1):
    labels = [lbl for _, lbl, _ in files]
    train_files, val_files = train_test_split(
        files,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return train_files, val_files


# =============================================================================
# NUMERICAL FEATURES (281)
# =============================================================================
def compute_statistics(feature_array):
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(1, -1)

    stats = []
    for row in feature_array:
        if len(row) > 0:
            stats.extend([np.mean(row), np.std(row), np.min(row), np.max(row)])
        else:
            stats.extend([0, 0, 0, 0])
    return np.array(stats)


def extract_all_281_features(y, sr=SR, verbose=False):
    if len(y) == 0:
        return np.zeros(281)

    features = []

    try:
        if verbose: print("  [1/16] MFCC Slaney...")
        try:
            mfcc_slaney = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, dct_type=2)
            features.append(compute_statistics(mfcc_slaney[:min(10, mfcc_slaney.shape[0])]))
        except:
            mfcc_slaney = None
            features.append(np.zeros(40))

        if verbose: print("  [2/16] MFCC HTK...")
        try:
            mfcc_htk = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, dct_type=3)
            features.append(compute_statistics(mfcc_htk[:min(10, mfcc_htk.shape[0])]))
        except:
            features.append(np.zeros(40))

        if verbose: print("  [3/16] Mel-Spectrogram...")
        try:
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            features.append(np.mean(mel_db, axis=1))
        except:
            features.append(np.zeros(128))

        if verbose: print("  [4/16] Chroma STFT...")
        try:
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, hop_length=1024)
            features.append(compute_statistics(chroma_stft))
        except:
            features.append(np.zeros(48))

        if verbose: print("  [5/16] Chroma CQT...")
        try:
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12, hop_length=2048, n_octaves=5)
            features.append(compute_statistics(chroma_cqt))
        except:
            try:
                chroma_fb = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
                features.append(compute_statistics(chroma_fb))
            except:
                features.append(np.zeros(48))

        if verbose: print("  [6/16] Chroma CENS...")
        try:
            chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=12, hop_length=1024)
            features.append(compute_statistics(chroma_cens))
        except:
            features.append(np.zeros(48))

        if verbose: print("  [7/16] Spectral Contrast...")
        try:
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
            features.append(compute_statistics(spec_contrast))
        except:
            features.append(np.zeros(28))

        if verbose: print("  [8/16] Spectral Flatness...")
        try:
            spec_flatness = librosa.feature.spectral_flatness(y=y)
            features.append(compute_statistics(spec_flatness))
        except:
            features.append(np.zeros(4))

        if verbose: print("  [9/16] Spectral Centroid...")
        try:
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(compute_statistics(spec_centroid))
        except:
            features.append(np.zeros(4))

        if verbose: print("  [10/16] Spectral Bandwidth...")
        try:
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features.append(compute_statistics(spec_bandwidth))
        except:
            features.append(np.zeros(4))

        if verbose: print("  [11/16] Spectral Rolloff...")
        try:
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.append(compute_statistics(spec_rolloff))
        except:
            features.append(np.zeros(4))

        if verbose: print("  [12/16] ZCR...")
        try:
            zcr = librosa.feature.zero_crossing_rate(y=y)
            features.append(compute_statistics(zcr))
        except:
            features.append(np.zeros(4))

        if verbose: print("  [13/16] Tonnetz Normal...")
        try:
            y_perc = librosa.effects.percussive(y, margin=0.5)
            tonnetz = librosa.feature.tonnetz(y=y_perc, sr=sr, hop_length=2048)
            features.append(compute_statistics(tonnetz))
        except:
            features.append(np.zeros(24))

        if verbose: print("  [14/16] Tonnetz Harmonic...")
        try:
            y_harm = librosa.effects.harmonic(y, margin=0.5)
            tonnetz_harm = librosa.feature.tonnetz(y=y_harm, sr=sr, hop_length=2048)
            features.append(compute_statistics(tonnetz_harm))
        except:
            features.append(np.zeros(24))

        if verbose: print("  [15/16] RMS Energy...")
        try:
            rms = librosa.feature.rms(y=y)
            features.append(compute_statistics(rms))
        except:
            features.append(np.zeros(4))

        if verbose: print("  [16/16] Delta MFCC...")
        try:
            if mfcc_slaney is not None:
                delta_mfcc = librosa.feature.delta(mfcc_slaney)
                features.append(compute_statistics(delta_mfcc))
            else:
                features.append(np.zeros(52))
        except:
            features.append(np.zeros(52))

        result = np.concatenate(features)
        if len(result) != 281:
            if len(result) < 281:
                result = np.pad(result, (0, 281 - len(result)), "constant")
            else:
                result = result[:281]
        return result

    except Exception as e:
        print(f"FATAL ERROR in feature extraction: {e}")
        return np.zeros(281)


def build_numerical_dataset(files, seg_dur):
    X, y, ids = [], [], []
    for fpath, label, _ in tqdm(files, desc=f"Numerical {seg_dur}s"):
        audio = load_audio(fpath)
        if audio is None or len(audio) == 0:
            continue
        segments = segment_audio(audio, seg_dur)
        for i, seg in enumerate(segments):
            if len(seg) == 0:
                continue
            X.append(extract_all_281_features(seg, SR, verbose=False))
            y.append(label)
            ids.append(f"{os.path.basename(fpath)}_seg{i}")
    return np.array(X), np.array(y), ids


# =============================================================================
# GRAPHICAL FEATURES
# =============================================================================
def _save_borderless(fig, save_path):
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(save_path, dpi=112, bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)


def create_spectrogram(y, sr, save_path):
    try:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_facecolor("black")
        D = librosa.stft(y)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(D_db, sr=sr, x_axis=None, y_axis=None, cmap="magma", ax=ax)
        ax.axis("off")
        _save_borderless(fig, save_path)
        return True
    except:
        plt.close("all")
        return False


def create_mel_spectrogram(y, sr, save_path):
    try:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_facecolor("black")
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None, cmap="magma", ax=ax)
        ax.axis("off")
        _save_borderless(fig, save_path)
        return True
    except:
        plt.close("all")
        return False


def create_mfcc_image(y, sr, save_path):
    try:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_facecolor("black")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfcc, sr=sr, x_axis=None, y_axis=None, cmap="viridis", ax=ax)
        ax.axis("off")
        _save_borderless(fig, save_path)
        return True
    except:
        plt.close("all")
        return False


def create_stft_image(y, sr, save_path):
    try:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_facecolor("black")
        D = librosa.stft(y)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(D_db, sr=sr, x_axis=None, y_axis=None, cmap="inferno", ax=ax)
        ax.axis("off")
        _save_borderless(fig, save_path)
        return True
    except:
        plt.close("all")
        return False


IMAGE_GENERATORS = {
    "spectrogram": create_spectrogram,
    "mel_spectrogram": create_mel_spectrogram,
    "mfcc": create_mfcc_image,
    "stft": create_stft_image,
}


def generate_graphical_split(files, split_name, max_per_class=None):
    """Generate 4 image types for each file into a split folder.

    The combined dataset is stored as:
      graphics_combined/<split_name>/HC/*.png
      graphics_combined/<split_name>/PD/*.png
    """
    split_root = os.path.join(GRAPHICS_COMBINED_DIR, split_name)
    for cls in ["HC", "PD"]:
        os.makedirs(os.path.join(split_root, cls), exist_ok=True)

    per_type_root = os.path.join(GRAPHICS_DIR, split_name)
    for img_type in IMAGE_GENERATORS:
        for cls in ["HC", "PD"]:
            os.makedirs(os.path.join(per_type_root, img_type, cls), exist_ok=True)

    counts = defaultdict(int)
    records = []

    for fpath, label, _ in tqdm(files, desc=f"Graphics {split_name}"):
        if max_per_class is not None and counts[label] >= max_per_class:
            continue

        audio = load_audio(fpath)
        if audio is None or len(audio) == 0:
            continue

        y_seg = audio[:5 * SR]
        label_name = CLASS_NAMES[label]
        file_id = os.path.basename(fpath).replace(".wav", "")

        for img_type, generator in IMAGE_GENERATORS.items():
            per_type_dir = os.path.join(per_type_root, img_type, label_name)
            combined_dir = os.path.join(split_root, label_name)
            img_name = f"{file_id}_{img_type}.png"
            per_type_path = os.path.join(per_type_dir, img_name)
            combined_path = os.path.join(combined_dir, img_name)

            ok = generator(y_seg, SR, per_type_path)
            if ok:
                shutil.copy2(per_type_path, combined_path)
                records.append({"filepath": combined_path, "label": label_name})

        counts[label] += 1

    return pd.DataFrame(records)


# =============================================================================
# NUMERICAL TRAIN / TEST WITHOUT CROSS-VALIDATION
# =============================================================================
def train_numerical_model(X_train, X_test, y_train, y_test, model_name, duration_tag):
    model_map = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "DT": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "NB": GaussianNB(),
    }

    clf = model_map[model_name]
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("selector", VarianceThreshold(threshold=0.0)),
        ("model", clf),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    model_dir = os.path.join(ML_MODELS_DIR, duration_tag)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(pipeline, model_path)

    report = classification_report(y_test, y_pred, target_names=["HC", "PD"], zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "duration": duration_tag,
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "model_path": model_path,
    }


def run_numerical_pipeline(train_files, test_files):
    print_section("NUMERICAL PIPELINE — 80/20 TRAIN / TEST")
    all_results = []

    for dur in SEGMENT_DURATIONS:
        print_subsection(f"{dur}s segments")

        X_train, y_train, ids_train = build_numerical_dataset(train_files, dur)
        X_test, y_test, ids_test = build_numerical_dataset(test_files, dur)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  Skipping {dur}s because data is missing.")
            continue

        # Save raw features and split manifests for reuse.
        np.savez(os.path.join(FEATURES_DIR, f"numerical_train_{dur}s.npz"), X=X_train, y=y_train, ids=ids_train)
        np.savez(os.path.join(FEATURES_DIR, f"numerical_test_{dur}s.npz"), X=X_test, y=y_test, ids=ids_test)

        for model_name in ["KNN", "SVM", "DT", "NB"]:
            result = train_numerical_model(X_train, X_test, y_train, y_test, model_name, f"{dur}s")
            all_results.append(result)
            print(f"  {model_name:<4} Acc={result['accuracy']:.4f}  F1={result['f1']:.4f}  Saved={result['model_path']}")

    if all_results:
        df = pd.DataFrame(all_results)
        out_csv = os.path.join(RESULTS_DIR, "numerical_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nSaved numerical results: {out_csv}")
        print(df[["duration", "model", "accuracy", "precision", "recall", "f1"]].to_string(index=False))

    return all_results


# =============================================================================
# GENERALIZED CNN MODELS ON COMBINED GRAPHICAL FEATURES
# =============================================================================
def build_cnn_model(model_name, input_shape=(100, 100, 3)):
    base_models = {
        "VGG19": VGG19,
        "ResNet50": ResNet50,
        "MobileNet": MobileNet,
    }
    base = base_models[model_name](weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )
    return model


def run_graphical_pipeline(train_files, test_files):
    print_section("GRAPHICAL PIPELINE — 3 GENERALIZED CNN MODELS")

    # Clean and build split folders so the split is file-level and reproducible.
    clean_directory(GRAPHICS_COMBINED_DIR)
    clean_directory(GRAPHICS_DIR)

    print_subsection("Generating validation images")
    # Validation comes from the training partition, not from the test partition.
    train_files2, val_files = split_train_val(train_files, val_size=0.1)
    clean_directory(os.path.join(GRAPHICS_COMBINED_DIR, "train"))
    clean_directory(os.path.join(GRAPHICS_COMBINED_DIR, "val"))
    clean_directory(os.path.join(GRAPHICS_COMBINED_DIR, "test"))
    clean_directory(os.path.join(GRAPHICS_DIR, "train"))
    clean_directory(os.path.join(GRAPHICS_DIR, "val"))
    clean_directory(os.path.join(GRAPHICS_DIR, "test"))

    train_df = generate_graphical_split(train_files2, "train", max_per_class=None)
    val_df = generate_graphical_split(val_files, "val", max_per_class=None)
    test_df = generate_graphical_split(test_files, "test", max_per_class=None)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("Not enough graphical data.")
        return None

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )
    val_gen = eval_datagen.flow_from_dataframe(
        val_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_dataframe(
        test_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    results = []
    for model_name in GRAPHICAL_CNN_MODELS:
        print_subsection(f"Training {model_name} on combined graphical data")

        model = build_cnn_model(model_name, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )

        test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen, verbose=0)
        test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec + 1e-8)
        y_score = model.predict(test_gen, verbose=0).flatten()
        y_true = test_gen.classes
        y_pred = (y_score >= 0.5).astype(int)

        model_path = os.path.join(MODELS_DIR, f"{model_name}_generalized_graphics.keras")
        model.save(model_path)

        metadata = {
            "model": model_name,
            "feature_mode": "combined_graphics",
            "image_types": ["spectrogram", "mel_spectrogram", "mfcc", "stft"],
            "img_size": IMG_SIZE,
            "class_indices": train_gen.class_indices,
        }
        save_json(os.path.join(MODELS_DIR, f"{model_name}_generalized_graphics.json"), metadata)

        report = classification_report(y_true, y_pred, target_names=["HC", "PD"], zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        result = {
            "model": model_name,
            "feature_mode": "combined_graphics",
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
            "test_loss": test_loss,
            "model_path": model_path,
            "report": report,
            "confusion_matrix": cm.tolist(),
            "history": history.history,
        }
        results.append(result)

        print(f"{model_name} test accuracy: {test_acc:.4f}")
        print(f"{model_name} test F1: {test_f1:.4f}")
        print(f"Saved model: {model_path}")

    out_csv = os.path.join(RESULTS_DIR, "graphical_results.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    print_section("FINAL PARKINSON PIPELINE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_DIR}")

    files = collect_files()
    if not files:
        print("ERROR: No audio files found.")
        return

    labels = [lbl for _, lbl, _ in files]
    n_hc = labels.count(0)
    n_pd = labels.count(1)
    print(f"Found {len(files)} files ({n_hc} HC, {n_pd} PD)")

    train_files, test_files = split_files(files, test_size=0.2)
    print(f"Train files: {len(train_files)} | Test files: {len(test_files)}")

    # Save split manifest for reproducibility.
    split_rows = []
    for path, lbl, cat in train_files:
        split_rows.append({"file": path, "label": lbl, "class": CLASS_NAMES[lbl], "split": "train", "category": cat})
    for path, lbl, cat in test_files:
        split_rows.append({"file": path, "label": lbl, "class": CLASS_NAMES[lbl], "split": "test", "category": cat})
    pd.DataFrame(split_rows).to_csv(os.path.join(RESULTS_DIR, "file_split_manifest.csv"), index=False)

    numerical_results = run_numerical_pipeline(train_files, test_files)
    graphical_results = run_graphical_pipeline(train_files, test_files)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_dir": DATASET_DIR,
        "train_files": len(train_files),
        "test_files": len(test_files),
        "numerical_models": len(numerical_results),
        "graphical_models": graphical_results,
    }
    save_json(os.path.join(RESULTS_DIR, "pipeline_summary.json"), summary)

    print_section("PIPELINE COMPLETE")
    print(f"Saved numerical models in: {ML_MODELS_DIR}")
    print(f"Saved CNN models in: {MODELS_DIR}")
    print(f"Saved numerical results in: {RESULTS_DIR}")
    print(f"Saved graphical results in: {RESULTS_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Final generalized Parkinson pipeline")
    parser.add_argument("--full", action="store_true", help="Run full 80/20 train/test pipeline")
    parser.add_argument("--test", action="store_true", help="Quick sanity check on one file")
    args = parser.parse_args()

    if args.test:
        files = collect_files()
        if not files:
            print("No files found.")
        else:
            fpath, label, _ = files[0]
            audio = load_audio(fpath)
            print(f"Test file: {fpath}")
            print(f"Audio length: {len(audio) / SR:.1f}s")
            feat = extract_all_281_features(audio[:5 * SR], SR, verbose=True)
            print(f"Numerical features: {feat.shape[0]}")
            print("Use --full for the complete train/test pipeline.")
    else:
        main()
