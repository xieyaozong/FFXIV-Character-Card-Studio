# ============================================================
# FFXIV Character Card Studio — Experiment Runner
# 修改下方區塊的參數後直接執行: .\run.ps1
# ============================================================

# ── 輸入圖片 ────────────────────────────────────────────────
$CharacterImage = "G:\path\to\character.jpg"   # 角色全身截圖
$WeaponImage    = "G:\path\to\weapon.png"       # 武器截圖

# ── 輸出目錄 ────────────────────────────────────────────────
$OutputDir = "outputs\experiments\run-001"      # 每次實驗換一個不同的名稱

# ── VLM 特徵 ────────────────────────────────────────────────
# 重用已有的 features.json 可跳過 Qwen，只調整生成參數
# 設為 "" 則重新執行 VLM 分析
$FeaturesFile = ""
# $FeaturesFile = "outputs\experiments\run-001\features.json"

# 只跑 VLM 輸出 features.json，不進行圖片生成（true/false）
$FeaturesOnly = $false

# ── 模型 ────────────────────────────────────────────────────
$VlmModel  = "models\vlm\Qwen3-VL-4B-Instruct"
# SDXL Base = 原始 baseline；Illustrious = 動漫風，需搭 Illustrious 系 LoRA
$SdxlModel = "models\diffusion\illustrious-xl-v0.1\Illustrious-XL-v0.1.safetensors"
# $SdxlModel = "models\diffusion\stable-diffusion-xl-base-1.0"

# ── LoRA（可加多個，格式: "路徑=權重"）──────────────────────
# Illustrious 系 LoRA 只能配 Illustrious base，勿與 SDXL Base 混用
$Loras = @(
    "models\loras\illustrious\style\keyframe-animation-v1.1.safetensors=0.65"   # style
    "models\loras\illustrious\ffxiv\au-ra-raen-facetype3-v2.safetensors=0.75"   # anatomy（角/鱗）
)

# ── 生成參數 ────────────────────────────────────────────────
$Seed          = 20260511
$Strength      = 0.46    # 0.40–0.70；越大越偏離輸入圖（開 IP-Adapter 後可拉高到 0.7–0.85 重繪）
$GuidanceScale = 6.5     # 典型值 5.0–8.0
$Steps         = 24      # 24–30

# ── IP-Adapter（身份注入：高 strength 重繪也保持臉是同一人）──
$IpAdapterScale = 0.0    # 0 = 關閉；建議 0.5–0.8。要「兩者兼顧」就開這個再把 $Strength 拉高
$IpAdapterImage = ""     # 留空 = 自動用角色臉部裁切；可指定一張臉部參考圖

# ── 完成度（hi-res 放大 + 臉部細修）─────────────────────────
$Hires           = $true   # 整張放大+輕重繪，提升解析度與完成度
$HiresScale      = 1.5     # 放大倍率（1.5 → 768x1024 變 1152x1536）
$HiresStrength   = 0.35    # 重繪強度（0.3–0.45；太高會偏離構圖）
$FaceDetail      = $true   # 頭部高解析重畫+貼回，修臉並強制長出種族解剖
$FaceDetailStrength = 0.5  # 臉部重繪強度（0.4–0.6）

# ── Prompt 微調 ─────────────────────────────────────────────
$ExtraPrompt    = ""     # 附加到 VLM 產生的 prompt 前面
$PromptOverride = ""     # 完全覆蓋 VLM prompt（留空則用 VLM 結果）
$ExtraNegative  = ""     # 附加到 negative prompt

# ── 背景移除（SDXL 初始圖用）────────────────────────────────
# 選項: "none" | "blue_screen" | "rembg"
# rembg 對任何背景都有效；blue_screen 只適合藍色沙龍照
$BackgroundBackend = "rembg"

# ── 主體裁切（送 VLM 前先框出角色放大，4K 截圖必開）─────────
$CropSubject = $true            # true/false
$CropBackend = "rembg"          # "rembg" | "blue_screen"
$CropPad     = 0.08             # 裁切框外擴比例

# ============================================================
# 以下不需要修改
# ============================================================

$args_list = @(
    "--character-image", $CharacterImage,
    "--weapon-image",    $WeaponImage,
    "--vlm-model",       $VlmModel,
    "--sdxl-model",      $SdxlModel,
    "--output-dir",      $OutputDir,
    "--seed",            $Seed,
    "--strength",        $Strength,
    "--guidance-scale",  $GuidanceScale,
    "--steps",           $Steps,
    "--background-backend", $BackgroundBackend,
    "--crop-backend",    $CropBackend,
    "--crop-pad",        $CropPad,
    "--ip-adapter-scale", $IpAdapterScale,
    "--hires-scale",      $HiresScale,
    "--hires-strength",   $HiresStrength,
    "--face-detail-strength", $FaceDetailStrength
)

if ($IpAdapterImage -ne "") {
    $args_list += "--ip-adapter-image", $IpAdapterImage
}

if ($Hires)      { $args_list += "--hires" }      else { $args_list += "--no-hires" }
if ($FaceDetail) { $args_list += "--face-detail" } else { $args_list += "--no-face-detail" }

if ($CropSubject) {
    $args_list += "--crop-subject"
} else {
    $args_list += "--no-crop-subject"
}

if ($FeaturesFile -ne "") {
    $args_list += "--features-file", $FeaturesFile
}

if ($FeaturesOnly) {
    $args_list += "--features-only"
}

if ($ExtraPrompt -ne "") {
    $args_list += "--extra-prompt", $ExtraPrompt
}

if ($PromptOverride -ne "") {
    $args_list += "--prompt-override", $PromptOverride
}

if ($ExtraNegative -ne "") {
    $args_list += "--extra-negative", $ExtraNegative
}

foreach ($lora in $Loras) {
    $args_list += "--lora", $lora
}

.\.venv\Scripts\Activate.ps1
python scripts\run_baseline_experiment.py @args_list
