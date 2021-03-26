English version below.

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

- Nahrát trénovací dataset na googledrive
- Trénink jsem provedl ve službě Google Colab
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gamecraftCZ/sofia-soc/blob/master/train.ipynb)
- ***TODO markdown to training iPython notebook***

### Test sítě

- Automatický test
    - `python simulator_drive_automatic_tests.py -m CESTA_K_MODELU`
- Manuální test
    - `python simulator_drive_manual-tests.py -m CESTA_K_MODELU`

---

# Sofia – neural network inspired by Caenorhabditis elegans and possibilities of its control

- ***TODO english***
