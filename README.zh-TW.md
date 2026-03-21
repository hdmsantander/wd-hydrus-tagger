[English](README.md)

# WD Tagger for Hydrus — 操作手冊

使用 WD14 Tagger (v3) 自動為 Hydrus Network 中的圖片產生標籤的 Web 工具。

---

## 目錄

- [系統需求](#系統需求)
- [安裝步驟](#安裝步驟)
- [Hydrus Network 前置設定](#hydrus-network-前置設定)
- [設定檔說明](#設定檔說明)
- [啟動伺服器](#啟動伺服器)
- [操作流程](#操作流程)
  - [Step 1：連線到 Hydrus](#step-1連線到-hydrus)
  - [Step 2：搜尋圖片](#step-2搜尋圖片)
  - [Step 3：選取圖片](#step-3選取圖片)
  - [Step 4：執行自動標記](#step-4執行自動標記)
  - [Step 5：檢視與編輯標記結果](#step-5檢視與編輯標記結果)
  - [Step 6：套用標籤到 Hydrus](#step-6套用標籤到-hydrus)
- [設定面板](#設定面板)
  - [模型管理](#模型管理)
  - [標籤前綴設定](#標籤前綴設定)
  - [GPU 加速](#gpu-加速)
- [可用模型一覽](#可用模型一覽)
- [閾值調整指南](#閾值調整指南)
- [設定檔參數完整說明](#設定檔參數完整說明)
- [常見問題](#常見問題)
- [專案結構](#專案結構)

---

## 系統需求

| 項目 | 要求 |
|------|------|
| Python | 3.10 以上 |
| Hydrus Network | 任何近期版本（需啟用 Client API） |
| 作業系統 | Windows / Linux / macOS |
| 硬碟空間 | 每個模型約 300–600 MB |
| 記憶體 | 建議 4 GB 以上（Large 模型建議 8 GB） |
| GPU（選用） | 支援 CUDA 的 NVIDIA 顯示卡 |

---

## 安裝步驟

### 1. 安裝 Python 依賴

```bash
# CPU 版（預設）
pip install -r requirements.txt

# GPU 版：onnxruntime-gpu 已內含 CPU 支援，直接替換即可
# 注意：兩者不能共存，需先移除 CPU 版
pip install -r requirements.txt --ignore-requires-python
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

### 2. 建立設定檔

```bash
# 將範例設定檔複製為實際設定檔
cp config.example.yaml config.yaml
```

然後編輯 `config.yaml`，填入你的 Hydrus API Key（詳見下方「Hydrus Network 前置設定」）。

---

## Hydrus Network 前置設定

使用本工具前，需要在 Hydrus Network 中啟用 Client API 並取得 API Key。

### 啟用 Client API

1. 打開 Hydrus Client
2. 前往選單 **services → manage services**
3. 確認 **client api** 服務已啟用（預設 port 45869）

### 取得 API Key

1. 前往選單 **services → review services → client api**
2. 點擊 **add → from api request**，或手動新增一組存取金鑰
3. 為這組金鑰命名（例如 `WD Tagger`）
4. 授予以下權限：
   - **search and fetch files**（搜尋與讀取檔案）
   - **import and edit tags**（匯入與編輯標籤）
5. 點擊 **apply** 取得一組 64 字元的十六進位金鑰
6. 將此金鑰貼入 `config.yaml` 的 `hydrus_api_key` 欄位

### （建議）建立專用 Tag Service

為了區隔 AI 生成的標籤與手動標籤，建議建立一個專用的本地 tag service：

1. 前往 **services → manage services**
2. 新增一個 **local tag service**，命名為例如 `AI Tags`
3. 在 `config.yaml` 中將 `target_tag_service` 設為 `"AI Tags"`

---

## 設定檔說明

設定檔為 `config.yaml`（YAML 格式）。首次使用請從 `config.example.yaml` 複製：

```yaml
# Hydrus Network 連線
hydrus_api_url: "http://localhost:45869"
hydrus_api_key: "你的64字元API金鑰"

# WD Tagger 設定
default_model: "wd-vit-tagger-v3"    # 預設使用的模型
models_dir: "./models"                # 模型存放目錄
use_gpu: false                        # 是否使用 GPU 加速

# 標籤閾值 (0.0 - 1.0)
general_threshold: 0.35               # 一般標籤的信心度閾值
character_threshold: 0.85             # 角色標籤的信心度閾值

# Hydrus tag service 名稱
target_tag_service: "my tags"

# 標籤前綴
general_tag_prefix: ""                # 一般標籤前綴（留空 = 無前綴）
character_tag_prefix: "character:"    # 角色標籤前綴
rating_tag_prefix: "rating:"          # 分級標籤前綴

# 批次處理
batch_size: 4                         # 每批處理的圖片數量

# Web UI 伺服器
host: "127.0.0.1"
port: 8199
```

---

## 啟動伺服器

```bash
python run.py
```

啟動後會看到：

```
INFO:     Uvicorn running on http://127.0.0.1:8199 (Press CTRL+C to quit)
```

打開瀏覽器前往 **http://127.0.0.1:8199** 即可使用。

---

## 操作流程

### Step 1：連線到 Hydrus

1. 打開瀏覽器前往 `http://127.0.0.1:8199`
2. 在左側邊欄的「Hydrus 連線」面板中：
   - **API URL**：填入 Hydrus Client API 的位址（預設 `http://localhost:45869`）
   - **API Key**：填入你的 API 金鑰
3. 點擊「連線」按鈕
4. 成功後，右上角狀態指示燈會從 **紅色** 變為 **綠色**
5. 連線資訊會自動儲存在瀏覽器的 localStorage 中，下次開啟時會自動連線

### Step 2：搜尋圖片

連線成功後，「搜尋圖片」面板會出現：

1. 在「搜尋標籤」輸入框中輸入搜尋條件
2. 使用逗號分隔多個條件
3. 點擊「搜尋」

**常用搜尋語法範例：**

| 語法 | 用途 |
|------|------|
| `system:archive` | 已歸檔的檔案 |
| `system:inbox` | 收件匣中的檔案 |
| `system:filetype is image` | 僅圖片檔案 |
| `system:archive, system:filetype is image` | 已歸檔的圖片 |
| `system:archive, -character:hatsune_miku` | 已歸檔但排除特定角色 |
| `system:filesize < 10MB` | 檔案大小小於 10MB |

搜尋完成後，下方會顯示找到的圖片數量。

### Step 3：選取圖片

搜尋結果會以縮圖 Grid 顯示在主內容區域：

- **單擊**：選取 / 取消選取單張圖片
- **Ctrl + 單擊**：多選（切換單張圖片的選取狀態）
- **Shift + 單擊**：範圍選取（選取兩次點擊之間的所有圖片）
- **「全選」按鈕**：選取當前頁面的所有圖片
- **「取消全選」按鈕**：清除所有選取

被選取的圖片會以紫色邊框標示，右上角顯示勾選標記。

頁面底部有分頁導覽，每頁顯示 50 張圖片。

### Step 4：執行自動標記

在左側邊欄的「Tagger 設定」面板中：

1. **模型**：從下拉選單選擇要使用的模型（首次使用時會自動下載）
2. **General 閾值**：拖動滑桿調整一般標籤的信心度門檻（預設 0.35）
3. **Character 閾值**：拖動滑桿調整角色標籤的信心度門檻（預設 0.85）
4. **Tag Service**：選擇要寫入的 Hydrus tag service
5. 點擊以下其中一個按鈕：
   - **「標記選取的圖片 (N)」**：只處理選取的圖片
   - **「標記全部搜尋結果」**：處理所有搜尋結果（超過 100 張會彈出確認對話框）

處理過程中會顯示進度條，包含目前進度和百分比。

> **首次使用注意**：模型檔案約 300–600 MB，首次載入會先從 HuggingFace 下載，需要一些時間。

### Step 5：檢視與編輯標記結果

標記完成後，畫面會切換到「標記結果」檢視：

每張圖片會顯示：
- **縮圖**
- **General 標籤**（藍色標籤）：一般描述性標籤，如 `1girl`、`blue hair`、`outdoors`
- **Character 標籤**（紫色標籤）：辨識出的角色，如 `hatsune miku`
- **Rating 標籤**（橘色標籤）：分級，如 `general`、`sensitive`

每個標籤都顯示信心度百分比。

**編輯功能：**
- 點擊標籤上的 **×** 按鈕可以移除不想要的標籤
- 移除後的標籤不會被套用到 Hydrus

### Step 6：套用標籤到 Hydrus

確認標記結果後：

1. 點擊「套用全部標籤到 Hydrus」按鈕
2. 標籤會寫入你在 Step 4 中選擇的 Tag Service
3. 完成後會顯示成功訊息

標籤會按照設定的前綴格式寫入：
- 一般標籤：`blue hair`（或加上自訂前綴）
- 角色標籤：`character:hatsune miku`
- 分級標籤：`rating:general`

點擊「返回圖片瀏覽」可以回到 Gallery 繼續處理其他圖片。

---

## 設定面板

點擊右上角的 **齒輪圖示（⚙）** 打開設定面板。

### 模型管理

顯示所有可用模型及其下載狀態：

- **已下載**：顯示綠色「已下載」文字
- **未下載**：顯示「下載」按鈕，點擊即可從 HuggingFace 下載

### 標籤前綴設定

| 欄位 | 說明 | 預設值 |
|------|------|--------|
| General 前綴 | 加在一般標籤前面的文字 | （空，即不加前綴） |
| Character 前綴 | 加在角色標籤前面的文字 | `character:` |
| Rating 前綴 | 加在分級標籤前面的文字 | `rating:` |

**自訂範例：**
- 想在所有標籤前加命名空間：General 前綴設為 `wd:` → 產生 `wd:blue hair`
- 不想加角色前綴：Character 前綴清空 → 產生 `hatsune miku`

### GPU 加速

勾選「使用 GPU 加速」可啟用 CUDA GPU 推論。需要：
- NVIDIA 顯示卡
- 已安裝 `onnxruntime-gpu`（`pip install onnxruntime-gpu`）

---

## 可用模型一覽

所有模型均來自 [SmilingWolf](https://huggingface.co/SmilingWolf) 的 WD Tagger v3 系列。

| 模型名稱 | 大小 | 說明 |
|----------|------|------|
| **WD ViT v3** | ~300 MB | 基礎 ViT 模型，速度與品質平衡，**推薦一般使用** |
| **WD SwinV2 v3** | ~300 MB | SwinTransformer V2 架構，與 ViT 品質相近 |
| **WD ViT Large v3** | ~600 MB | 大型 ViT 模型，精度更高但較慢 |
| **WD EVA02 Large v3** | ~600 MB | EVA02 架構大型模型，精度最高 |

**建議：**
- 一般用途、批次處理大量圖片 → **WD ViT v3** 或 **WD SwinV2 v3**
- 追求最高精度 → **WD EVA02 Large v3**

---

## 閾值調整指南

閾值（threshold）決定了多少信心度以上的標籤會被保留。

### General 閾值（預設 0.35）

| 閾值範圍 | 效果 |
|----------|------|
| **0.20–0.30** | 寬鬆，會產生較多標籤，可能包含不精確的 |
| **0.35–0.45** | 平衡，適合大多數用途 |
| **0.50–0.70** | 嚴格，只保留高信心度的標籤 |
| **0.70+** | 非常嚴格，只保留最明顯的特徵 |

### Character 閾值（預設 0.85）

角色辨識建議使用較高的閾值以避免誤判：

| 閾值範圍 | 效果 |
|----------|------|
| **0.70–0.80** | 較寬鬆，可能出現誤判角色 |
| **0.85–0.90** | 平衡，適合大多數情況 |
| **0.90+** | 只有非常確定的角色才會被標記 |

---

## 設定檔參數完整說明

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `hydrus_api_url` | string | `http://localhost:45869` | Hydrus Client API 位址 |
| `hydrus_api_key` | string | `""` | 64 字元 API 金鑰 |
| `default_model` | string | `wd-vit-tagger-v3` | 預設載入的模型名稱 |
| `models_dir` | string | `./models` | 模型檔案儲存目錄 |
| `use_gpu` | bool | `false` | 是否使用 CUDA GPU |
| `general_threshold` | float | `0.35` | General 標籤閾值 |
| `character_threshold` | float | `0.85` | Character 標籤閾值 |
| `target_tag_service` | string | `my tags` | 預設寫入的 tag service |
| `general_tag_prefix` | string | `""` | General 標籤前綴 |
| `character_tag_prefix` | string | `character:` | Character 標籤前綴 |
| `rating_tag_prefix` | string | `rating:` | Rating 標籤前綴 |
| `batch_size` | int | `4` | 每批推論的圖片數 |
| `host` | string | `127.0.0.1` | Web UI 綁定位址 |
| `port` | int | `8199` | Web UI 埠號 |

---

## 常見問題

### Q：連線失敗，顯示紅燈

- 確認 Hydrus Client 正在運行
- 確認 Client API 服務已啟用（services → manage services）
- 確認 API URL 和 port 正確（預設 45869）
- 確認 API Key 正確且有足夠權限

### Q：模型下載很慢或失敗

- 模型從 HuggingFace 下載，需要穩定的網路連線
- Large 模型約 600 MB，請耐心等待
- 如果下載失敗，可以手動從 HuggingFace 下載 `model.onnx` 和 `selected_tags.csv` 放到 `models/{模型名稱}/` 目錄下
- 手動下載網址範例：`https://huggingface.co/SmilingWolf/wd-vit-tagger-v3`

### Q：標記速度很慢

- 使用較小的模型（ViT v3 或 SwinV2 v3）
- 減少 `batch_size`（降低記憶體壓力）
- 啟用 GPU 加速（需 NVIDIA 顯卡 + `onnxruntime-gpu`）

### Q：標籤太多 / 太少

- 調整 General 閾值：調高 = 標籤更少但更精確，調低 = 標籤更多但可能不精確
- 在套用前可以手動移除不需要的標籤

### Q：角色辨識不準確

- 調高 Character 閾值（例如 0.90 以上）
- 使用 Large 模型以提高辨識精度
- 注意：模型只能辨識訓練資料中包含的角色（主要來自 Danbooru）

### Q：如何只處理特定類型的檔案？

在搜尋欄位中加入檔案類型篩選：
- `system:archive, system:filetype is image` — 只搜尋圖片
- `system:archive, system:filetype is png` — 只搜尋 PNG
- `system:archive, system:width > 512` — 只搜尋寬度大於 512 的

### Q：標籤中的底線怎麼處理？

工具會自動將底線 `_` 轉換為空格。例如模型輸出 `blue_hair` 會變成 `blue hair`。

### Q：能否遠端使用？

將 `config.yaml` 中的 `host` 改為 `0.0.0.0` 即可接受外部連線。但請注意：
- 本工具沒有內建認證機制
- 不建議暴露在公共網路上
- 如需遠端使用，建議透過 SSH tunnel 或 VPN

---

## 專案結構

```
wd-hydrus-tagger/
├── run.py                    # 啟動入口
├── config.yaml               # 設定檔（需自行建立）
├── config.example.yaml       # 設定檔範例
├── pyproject.toml             # Python 專案設定
├── backend/                   # FastAPI 後端
│   ├── app.py                 #   應用程式主體
│   ├── config.py              #   設定檔載入
│   ├── dependencies.py        #   依賴注入
│   ├── hydrus/                #   Hydrus API 客戶端
│   │   ├── client.py          #     非同步 HTTP 客戶端
│   │   └── models.py          #     資料模型
│   ├── tagger/                #   WD Tagger 推論引擎
│   │   ├── engine.py          #     ONNX 推論
│   │   ├── preprocess.py      #     圖片前處理
│   │   ├── labels.py          #     標籤詞表解析
│   │   └── models.py          #     結果資料模型
│   ├── routes/                #   API 路由
│   │   ├── connection.py      #     連線管理
│   │   ├── files.py           #     檔案瀏覽
│   │   ├── tagger.py          #     標記功能
│   │   └── config_routes.py   #     設定管理
│   └── services/              #   服務層
│       ├── tagging_service.py #     批次標記協調器
│       └── model_manager.py   #     模型下載管理
├── frontend/                  # Web 前端
│   ├── index.html             #   單頁應用程式
│   ├── css/style.css          #   樣式表（暗色主題）
│   └── js/                    #   JavaScript 模組
│       ├── app.js             #     應用程式初始化
│       ├── api.js             #     後端 API 客戶端
│       ├── state.js           #     狀態管理
│       ├── components/        #     UI 元件
│       └── utils/             #     工具函式
└── models/                    # 模型檔案（自動建立）
```
