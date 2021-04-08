English version follows.

# Sofie – neuronová síť inspirovaná háďátkem obecným a možnosti její kontroly

- Tento repozitář obsahuje kódy k replikaci experimentů, které jsem dělal v rámci
  mé [Středoškolské Odborné Činnosti](https://www.soc.cz).
- Text práce je součástí repozitáře: [SOCka_SofiaAI_Patrik_Vacal.pdf](SOCka_SofiaAI_Patrik_Vacal.pdf)
- Upravená verze PGDrive je k dispozici na mém
  githubu: [https://github.com/gamecraftCZ/pgdrive](https://github.com/gamecraftCZ/pgdrive)

### Příprava prostředí

- K experimentu byl využit python 3.7 (verze pythonu využívaná na Google Colabu), doporučuji využít tu.
- Instalace potřebných knihoven:
    - CPU: `pip install -r requirements-cpu.txt`
    - GPU: `pip install -r requirements-gpu.txt`
- Při použití verze GPU je nutná instalace CUDA 11.0 a cuDNN 8.0

### Sběr trénovacích dat

- Automatický sběr
    - `python collect_data.py -m a`
- Manuální sběr
    - `python collect_data.py -m k`

### Trénink sítě

- Vytvořený dataset je třeba nahrát na Google Drive
- Trénink jsem provedl ve službě Google Colab (train.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gamecraftCZ/sofia-soc/blob/master/train.ipynb)

### Test sítě

- Automatický test
    - `python simulator_drive_automatic_tests.py -m <CESTA_K_MODELU>`
- Manuální test
    - `python simulator_drive_manual-tests.py -m <CESTA_K_MODELU>`

![Video jízdy](sofia_drive-speed-arrows.gif)

### Řešení možných problémů

- PGDrive hlásí `UnicodeDecodeError: 'cp950' codec can't decode byte 0x8d in position 49100: illegal multibyte sequence`
    - Řešení: Přepnout Windows encoding do módu UTF-8
    - Github issue: https://github.com/decisionforce/pgdrive/issues/251

### BONUS: Řízení pomocí hlasu

1. Spustíme simulátor
    - `python simulator_drive_siri.py -m <CESTA_K_MODELU>`
2. Pro IOS nastavíme Shortcut tak, aby poslal request na: `http://<ip_počítače>:<port>/<zatočení>`
    - Např: `http://192.198.101.61:2002/left`
3. Užívejte jízdu!
    - Ukázka: https://youtu.be/6lJG2cZftH8

---

# Sofia – neural network inspired by Caenorhabditis elegans and possibilities of its control

- This repository contains codes to replicate experiments I did as part of
  my [Students` Professional Activities](https://www.soc.cz/english/).
- Text version of my work (CZ) is part of this
  repository: [SOCka_SofiaAI_Patrik_Vacal.pdf](SOCka_SofiaAI_Patrik_Vacal.pdf)
- Forked version of PGDrive used in these experiments is available on my
  github: [https://github.com/gamecraftCZ/pgdrive](https://github.com/gamecraftCZ/pgdrive)

### Experiment environment

- I used python 3.7 (version used in Google Colab), so I recommend using the same.
- Installation of required libraries:
    - CPU: `pip install -r requirements-cpu.txt`
    - GPU: `pip install -r requirements-gpu.txt`
- When using GPU version, installation of CUDA 11.0 and cuDNN 8.0 is required too.

### Training data collection

- Automatic collection
    - `python collect_data.py -m a`
- Manual collection
    - `python collect_data.py -m k`

### Neural network training

- Created dataset has to be uploaded to Google Drive
- I used Google Colab for training the network (train.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gamecraftCZ/sofia-soc/blob/master/train.ipynb)

### Test of trained network

- Automatic test
    - `python simulator_drive_automatic_tests.py -m <PATH_TO_MODEL>`
- Manual test
    - `python simulator_drive_manual-tests.py -m <PATH_TO_MODEL>`

![Ride video](sofia_drive-speed-arrows.gif)

### Troubleshooting

- PGDrive crashes
  with `UnicodeDecodeError: 'cp950' codec can't decode byte 0x8d in position 49100: illegal multibyte sequence`
    - Solution: Switch Windows encoding to UTF-8
    - See Github issue: https://github.com/decisionforce/pgdrive/issues/251

### BONUS: Voice control

1. Start simulator
    - `python simulator_drive_siri.py -m <PATH_TO_MODEL>`
2. For IOS create Shortcut that sends request to: `http://<computer_ip>:<port>/<turn>`
    - eg. `http://192.198.101.61:2002/left`
3. Enjoy ride!
    - Demonstration: https://youtu.be/6lJG2cZftH8
  
