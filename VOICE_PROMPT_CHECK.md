# 語音提示功能與程式檢查紀錄

## 新增功能

在右側設定面板的「顯示設定」下方新增「啟用語音提示」勾選項目。

勾選後，當計數事件發生時：

- 含 B 的物件通過：播放 `B.wav`
- 無 B 的物件通過：播放 `NO_B.wav`

語音播放採用非同步方式，不會等待音檔播放完才繼續推論。

## 音檔放置位置

請將音檔放在專案根目錄：

```text
yolo_counter-008/
  B.wav
  NO_B.wav
```

如果勾選語音提示但找不到音檔，程式不會中斷，會在狀態列提示找不到對應音檔。

## 修改檔案

- `config.py`
  - 新增 `voice_prompt_enabled` 設定欄位，讓語音提示開關可儲存。

- `gui/control_panel.py`
  - 在「顯示設定」新增「啟用語音提示」勾選項目。
  - 勾選狀態會同步到 `AppSettings`。
  - 新增 `voice_prompt_changed` signal，讓主視窗可即時接收開關狀態。

- `gui/main_window.py`
  - 接收語音提示開關狀態。
  - 在每次 `CountEvent` 發生後依 `has_b` 播放 `B.wav` 或 `NO_B.wav`。
  - 使用 Windows `winsound.PlaySound(..., SND_ASYNC)` 非同步播放音檔。

## 觸發邏輯

語音播放只會在「新的計數事件」發生時觸發，不會因為畫面持續偵測到同一個物件而重複播放。

依據欄位：

```text
CountEvent.has_b == True   -> B.wav
CountEvent.has_b == False  -> NO_B.wav
```

## 程式檢查結果

已對目前專案所有主要 Python 檔案執行語法編譯檢查：

```powershell
python -m py_compile mainAPP.py config.py core\video_source.py core\inference.py core\counter.py core\csv_logger.py gui\main_window.py gui\control_panel.py gui\video_widget.py gui\roi_editor.py gui\dashboard.py
```

檢查結果：通過。

備註：目前專案入口檔為 `mainAPP.py`，不是 `main.py`。
