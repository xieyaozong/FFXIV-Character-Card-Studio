# Manual Experiments

Use this when tuning locally from VS Code PowerShell.

## Run

```powershell
python scripts\run_baseline_experiment.py `
  --character-image "character\ffxiv_20260511_201318_995.png" `
  --weapon-image "character\ffxiv_20260511_201348_157.png" `
  --output-dir "outputs\experiments\manual-001" `
  --features-only
```

Run the full local generation path after the model files are in place:

```powershell
python scripts\run_baseline_experiment.py `
  --character-image "character\ffxiv_20260511_201318_995.png" `
  --weapon-image "character\ffxiv_20260511_201348_157.png" `
  --output-dir "outputs\experiments\manual-001"
```

## Edit

Change only the CLI values you are testing: image paths, output folder, seed, steps, strength, guidance scale, LoRA
arguments, or extra prompt text.

## Fast Loop

Run feature extraction once:

```powershell
$FeaturesOnly = $true
$FeaturesFile = ""
```

Reuse the saved features for generation tests:

```powershell
$FeaturesOnly = $false
$FeaturesFile = "outputs\experiments\<old-run>\features.json"
$OutputName = "manual-002"
```

Change only generation knobs:

```powershell
$Strength = 0.46
$GuidanceScale = 6.5
$Steps = 24
$Seed = 20260511
```

Results are written to:

```text
outputs\experiments\<OutputName>\
```

## Knowledge Data

Edit the local knowledge files directly:

```text
knowledge\ffxiv\anatomy_rules.yaml
knowledge\ffxiv\entities.yaml
knowledge\ffxiv\race_signatures.yaml
knowledge\ffxiv\special_features.yaml
```

Then run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_content_pack.py -q -p no:cacheprovider
```
