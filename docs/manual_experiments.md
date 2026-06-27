# Manual Experiments

Use this when tuning locally from VS Code PowerShell.

## Run

```powershell
cd G:\side_project\ffxiv-character-card-studio
.\.venv\Scripts\Activate.ps1
.\run.ps1
```

Preview the command without running models:

```powershell
.\run.ps1 -DryRun
```

## Edit Here

Open `run.ps1` and edit only the first block:

- `$CharacterImage`
- `$WeaponImage`
- `$OutputName`
- `$FeaturesFile`
- `$FeaturesOnly`
- `$Seed`
- `$Strength`
- `$GuidanceScale`
- `$Steps`
- `$ExtraPrompt`
- `$PromptOverride`
- `$ExtraNegative`
- `$Loras`

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

Edit the private files directly:

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
