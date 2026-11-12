#
# USPOREDNI MODELI – unaprijed zapisani u dnevniku eksperimenata (11. 11. 2026.)
# Isti podaci (data_split), ista augmentacija, iste težine klasa, 10 epoha, najbolja epoha po validacijskom AUC-u.
#   zamrznuti  : ResNet-18 sa zamrznutim slojevima – uči se SAMO zadnji sloj (recept DL17/DL18), lr 1e-3
#   seed_123   : isti model kao glavni, samo drugi seed treninga (podjela podataka ostaje ista, seed 42)
#   seed_2024  : isto, seed 2024
# Pokretanje:  python train_usporedba.py zamrznuti          (pravo treniranje)
#              python train_usporedba.py zamrznuti probni   (kratki probni pokretaj)
# Glavni model (runs/melanom_resnet18) se NE dira.
#
import sys                                                # čitanje imena varijante iz naredbenog retka
import time                                               # mjerenje trajanja epohe
from pathlib import Path                                  # putanje

import torch                                              # PyTorch
import torch.nn as nn                                     # funkcija gubitka
from torch.utils.tensorboard import SummaryWriter         # zapis tijeka učenja

# sve ostalo uzimamo iz glavne skripte – isti podaci, ista augmentacija, isti model, ista validacija
from train_melanom import (set_seed, make_dataloaders_with_imagefolder, build_model,
                           evaluate_val, LEARNING_RATE, LOG_EVERY)


#
# GLOBALNE POSTAVKE
#
VARIJANTE = {                                   # ime varijante -> (seed treninga, stopa učenja, zamrznuti?)
    "zamrznuti": (42, 1e-3, True),              # samo zadnji sloj; veća stopa učenja jer uči samo 1026 težina
    "seed_123": (123, LEARNING_RATE, False),    # kao glavni model (lr 1e-4), drugi seed
    "seed_2024": (2024, LEARNING_RATE, False),  # kao glavni model, treći seed
}
EPOCHS = 10                                     # kao glavni model


def main() -> None:
    varijanta = sys.argv[1]                                     # npr. "zamrznuti"
    probni = len(sys.argv) > 2 and sys.argv[2] == "probni"    # True ako je dodano "probni"
    seed, lr, zamrznuti = VARIJANTE[varijanta]                  # postavke te varijante
    epohe = 1 if probni else EPOCHS                             # probni rad: 1 epoha
    max_batcheva = 30 if probni else None                       # probni rad: 30 batcheva

    run_dir = Path("runs") / ("usporedba_" + varijanta + ("_probni" if probni else ""))   # zasebna mapa
    set_seed(seed)                                                         # seed treninga
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Varijanta:", varijanta, "| seed:", seed, "| lr:", lr, "| zamrznuti:", zamrznuti, "| probni:", probni)

    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    train_loader, val_loader, classes, class_weights = make_dataloaders_with_imagefolder()
    model = build_model(num_classes=len(classes)).to(device)               # ResNet-18, novi zadnji sloj

    if zamrznuti:                                       # recept s predavanja DL17/DL18
        for param in model.parameters():
            param.requires_grad = False                 # zamrzni sve težine
        for param in model.fc.parameters():
            param.requires_grad = True                  # odmrzni samo zadnji sloj (512 -> 2)

    ucenje = [p for p in model.parameters() if p.requires_grad]            # težine koje se uče
    print("Broj težina koje se uče:", sum(p.numel() for p in ucenje))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))       # iste težine klasa
    optimizer = torch.optim.Adam(ucenje, lr=lr)                            # Adam samo za težine koje se uče

    global_step = 0
    best_val_auc = -1.0
    for epoch in range(1, epohe + 1):
        model.train()
        running_loss = 0.0
        start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Pet koraka učenja (isto kao glavni model):
            optimizer.zero_grad(set_to_none=True)   # 1. obriši stare gradijente
            logits = model(x)                       # 2. predviđanje
            loss = criterion(logits, y)             # 3. greška
            loss.backward()                         # 4. backpropagation
            optimizer.step()                        # 5. ažuriraj težine

            running_loss += loss.item()
            global_step += 1
            if batch_idx % LOG_EVERY == 0:
                writer.add_scalar("train/loss", running_loss / LOG_EVERY, global_step)
                running_loss = 0.0
            if max_batcheva is not None and batch_idx >= max_batcheva:     # probni rad: prekini
                break

        val = evaluate_val(model, val_loader, device)                      # validacija nakon epohe
        writer.add_scalar("val/auc", val["auc"], epoch)
        print(f"Epoch {epoch:02d}/{epohe} | val_AUC={val['auc']:.4f} | "
              f"osjetljivost@0.5={val['osjetljivost']:.3f} | specifičnost@0.5={val['specificnost']:.3f} | "
              f"{(time.time() - start) / 60:.1f} min")

        if val["auc"] >= best_val_auc:                                     # isto pravilo kao glavni model
            best_val_auc = val["auc"]
            torch.save({"model_state_dict": model.state_dict(), "num_classes": len(classes),
                        "classes": classes, "epoch": epoch, "val_auc": best_val_auc},
                       run_dir / "best_model.pt")
            print(f"   -> spremljen najbolji model (val_AUC={best_val_auc:.4f})")

    writer.close()
    print(f"Gotovo: {varijanta} | best val_AUC={best_val_auc:.4f} | model: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
