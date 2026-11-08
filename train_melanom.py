#
# Trening modela za prepoznavanje melanoma – transfer learning s ResNet-18
# Tijek: slike iz data_split/train i data_split/val (već podijeljene PO LEZIJI)
# - augmentacija samo u treningu - ResNet-18, novi zadnji sloj 512 -> 2
# - gubitak s težinama klasa (melanoma je ~8x manje) - nakon svake epohe validacija:
#   AUC, osjetljivost i specifičnost - spremam model s NAJBOLJIM validacijskim AUC-om
#
import time                                               # mjerenje trajanja epohe
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score                 # AUC iz vjerojatnosti (ML5)


#
# GLOBALNE POSTAVKE
#
PROBNI_RAD = False          # True = kratki probni pokretaj; False = pravo treniranje

TRAIN_DIR = Path("data_split") / "train"
VAL_DIR = Path("data_split") / "val"
RUN_DIR = Path("runs") / ("melanom_probni" if PROBNI_RAD else "melanom_resnet18")

EPOCHS = 1 if PROBNI_RAD else 10           # broj prolazaka kroz sve trening slike
MAX_BATCHES = 30 if PROBNI_RAD else None   # probni rad: samo prvih 30 batcheva
BATCH_SIZE = 32
LEARNING_RATE = 1e-4       # mala stopa učenja: fino podešavanje već naučene mreže
NUM_WORKERS = 4            # pomoćni procesi za učitavanje slika
SEED = 42
LOG_EVERY = 10             # svakih 10 batcheva zapiši gubitak u TensorBoard
MELANOM = 1                # indeks klase melanom (mapa 1_melanom)


def set_seed(seed: int):
    torch.manual_seed(seed)              # iste početne težine novog sloja i isto miješanje slika
    torch.cuda.manual_seed_all(seed)


def make_dataloaders_with_imagefolder() -> Tuple[DataLoader, DataLoader, list, torch.Tensor]:
    for mapa in [TRAIN_DIR, VAL_DIR]:
        if not mapa.exists():
            raise FileNotFoundError(f"Ne nalazim: {mapa} – prvo pokreni podijeli_ham.py")

    weights = ResNet18_Weights.DEFAULT

    # Trening: augmentacija – lezija nema "gore", a boja je dijagnostički važna
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),   # nasumičan izrez i povećanje
        transforms.RandomHorizontalFlip(p=0.5),                 # zrcaljenje lijevo-desno
        transforms.RandomVerticalFlip(p=0.5),                   # zrcaljenje gore-dolje
        transforms.RandomRotation(30),                          # zakretanje do 30 stupnjeva
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),  # blago: razlike kamera
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])

    # Validacija: bez augmentacije; Resize(256) + CenterCrop(224) čuva omjer stranica
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])

    # Skupovi su već podijeljeni po leziji u zasebne mape – zato nema random_split
    train_ds = ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    val_ds = ImageFolder(root=str(VAL_DIR), transform=val_transform)

    print("Klase:", train_ds.class_to_idx)            # očekujem {'0_ostalo': 0, '1_melanom': 1}
    if train_ds.class_to_idx["1_melanom"] != MELANOM:
        raise ValueError("Melanom nema indeks 1!")

    # Težine klasa – SAMO iz trening skupa: n / (broj_klasa × n_klase)  (DL6, ML8)
    n_po_klasi = torch.bincount(torch.tensor(train_ds.targets))       # [broj ostalo, broj melanom]
    class_weights = len(train_ds) / (len(n_po_klasi) * n_po_klasi.float())
    print("Trening slike po klasi:", n_po_klasi.tolist(), "| težine klasa:", class_weights)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin)

    print(f"Split: train={len(train_ds)}, val={len(val_ds)}")
    return train_loader, val_loader, train_ds.classes, class_weights


def build_model(num_classes: int) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)                          # mreža naučena na ImageNetu
    model.fc = nn.Linear(model.fc.in_features, num_classes)   # novi zadnji sloj: 512 -> 2
    return model                                               # ništa nije zamrznuto: uči se cijela mreža


@torch.no_grad()
def evaluate_val(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    # AUC te osjetljivost i specifičnost pri pragu 0,5 (pravi prag biramo u evaluate_melanom.py)
    model.eval()
    sve_vjerojatnosti = []
    sve_oznake = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)                                   # 2 broja po slici
        p_mel = torch.softmax(logits, dim=1)[:, MELANOM]    # vjerojatnost melanoma (0–1)
        sve_vjerojatnosti.extend(p_mel.cpu().tolist())
        sve_oznake.extend(y.tolist())

    p = torch.tensor(sve_vjerojatnosti)
    y = torch.tensor(sve_oznake)
    pred = (p >= 0.5).long()                               # odluka pri pragu 0,5

    tp = ((pred == 1) & (y == 1)).sum().item()             # melanom -> model kaže melanom
    fn = ((pred == 0) & (y == 1)).sum().item()             # melanom -> model kaže ostalo (PROPUŠTEN)
    tn = ((pred == 0) & (y == 0)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()

    return {
        "auc": roc_auc_score(sve_oznake, sve_vjerojatnosti),
        "osjetljivost": tp / (tp + fn),
        "specificnost": tn / (tn + fp),
    }


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| PROBNI_RAD:", PROBNI_RAD)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(RUN_DIR / "tb"))

    train_loader, val_loader, classes, class_weights = make_dataloaders_with_imagefolder()

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))   # propušten melanom "boli" ~8x više
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    global_step = 0
    best_val_auc = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Pet koraka učenja za jedan batch:
            optimizer.zero_grad(set_to_none=True)   # 1. obriši stare gradijente
            logits = model(x)                       # 2. predviđanje
            loss = criterion(logits, y)             # 3. greška (s težinama klasa)
            loss.backward()                         # 4. backpropagation
            optimizer.step()                        # 5. Adam ažurira težine

            running_loss += loss.item()
            global_step += 1

            if batch_idx % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                writer.add_scalar("train/loss", avg_loss, global_step)
                print(f"   batch {batch_idx}/{len(train_loader)} | loss={avg_loss:.4f}")
                running_loss = 0.0

            if MAX_BATCHES is not None and batch_idx >= MAX_BATCHES:   # probni rad: prekini ranije
                break

        val = evaluate_val(model, val_loader, device)
        trajanje = (time.time() - start) / 60

        writer.add_scalar("val/auc", val["auc"], epoch)
        writer.add_scalar("val/osjetljivost_0.5", val["osjetljivost"], epoch)
        writer.add_scalar("val/specificnost_0.5", val["specificnost"], epoch)

        print(f"Epoch {epoch:02d}/{EPOCHS} | val_AUC={val['auc']:.4f} | "
              f"osjetljivost@0.5={val['osjetljivost']:.3f} | specifičnost@0.5={val['specificnost']:.3f} | "
              f"{trajanje:.1f} min")

        if val["auc"] >= best_val_auc:     # najbolji model biramo po AUC-u, NE po točnosti
            best_val_auc = val["auc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(classes),
                    "classes": classes,              # ['0_ostalo', '1_melanom']
                    "epoch": epoch,
                    "val_auc": best_val_auc,
                },
                RUN_DIR / "best_model.pt",
            )
            print(f"   -> spremljen najbolji model (val_AUC={best_val_auc:.4f})")

    writer.close()
    print(f"Trening gotov. Best val_AUC={best_val_auc:.4f}")
    print(f"Model: {RUN_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
