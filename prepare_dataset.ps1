# ============================================================
# FFXIV Character Card Studio — Per-character LoRA dataset prep
# 修改下方參數後直接執行: .\prepare_dataset.ps1
# 把多角度角色截圖整理成 kohya 訓練用資料集（CPU，可重複執行）
# ============================================================

# ── 輸入：多角度參考截圖資料夾 ──────────────────────────────
$InputDir = "private_inputs\my_character\refs"

# ── 輸出：資料集根目錄 ──────────────────────────────────────
$OutputDir = "datasets\my_character"

# ── 觸發詞（要獨特，避免撞到底模詞彙；建議用角色名）─────────
$Trigger = "myocname"

# ── 說明標籤 ────────────────────────────────────────────────
$ClassTag   = "1girl"   # 每張 caption 附加的類別標籤；留空則只有觸發詞
$Repeats    = 10        # kohya 每張重複次數（資料夾名前綴）
$Resolution = 1024      # 最長邊縮放目標
$CropPad    = 0.12      # 主體裁切外擴比例

# ── 選項 ────────────────────────────────────────────────────
$CropSubject = $true    # 先框出角色再縮放
$TriageSkip  = $false   # true = 自動丟掉太暗/太糊的（預設保留，因為是你精選的）

# ============================================================
# 以下不需要修改
# ============================================================
$args_list = @(
    "--input-dir",  $InputDir,
    "--output-dir", $OutputDir,
    "--trigger",    $Trigger,
    "--class-tag",  $ClassTag,
    "--repeats",    $Repeats,
    "--resolution", $Resolution,
    "--crop-pad",   $CropPad
)
if ($CropSubject) { $args_list += "--crop-subject" } else { $args_list += "--no-crop-subject" }
if ($TriageSkip)  { $args_list += "--triage-skip" }

.\.venv\Scripts\Activate.ps1
python scripts\prepare_character_dataset.py @args_list
