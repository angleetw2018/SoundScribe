# 聲音筆記 SoundScribe

Windows 10/11 本機電腦聲音逐字稿工具。它透過 WASAPI loopback 監聽目前喇叭或耳機正在播放的聲音，再以 Faster-Whisper 在電腦上辨識；除了首次下載模型外，不會把錄音上傳到雲端，也不需要 API Key。

## 安裝與啟動

1. 安裝 64 位元 Python 3.11 或 3.12，安裝時勾選 `Add Python to PATH`。
2. 雙擊 `install.bat`，等待套件安裝完成。
3. 雙擊 `start.bat`。
4. 選擇目前播放聲音的輸出裝置、模型與語言，按「開始監聽」。

第一次使用某個模型時會自動下載。CPU 建議先用 `small`；有 NVIDIA GPU 可嘗試 `turbo` 或 `large-v3`。模型越大越準，但需要更多記憶體。

## 輸出

每次開始監聽都會在 `transcripts` 資料夾建立帶時間戳的 UTF-8 `.txt` 檔，停止前也會持續寫入，意外關閉時已辨識的內容仍會保留。

## 常見問題

- **沒有裝置**：確認 Windows 聲音設定中選定的喇叭／耳機目前為啟用狀態，然後按「重新整理裝置」。
- **辨識很慢**：改用 `tiny` 或 `base`；關閉其他占用 GPU／CPU 的程式。
- **缺少 cublas／cuDNN DLL**：程式會自動切換至 CPU 並重試，不需要另外安裝 CUDA。若想使用 NVIDIA GPU 加速，再另行安裝與 CTranslate2 相容的 CUDA 及 cuDNN。
- **沒有文字**：確認選的是實際播放聲音的裝置，並提高系統播放音量。
- **首次啟動很久**：正在下載模型，等待完成即可；之後會使用快取。

## 設計參考

- [OpenAI Whisper](https://github.com/openai/whisper)：核心語音辨識設計參考，約 11 萬 GitHub stars（查詢於 2026-10-14）。
- [realtime-captions-system-audio](https://github.com/emidium-science/realtime-captions-system-audio)：Windows WASAPI loopback 即時字幕流程參考。

## 隱私提醒

錄製會議或通話前，請先取得參與者同意並遵守所在地法律與公司政策。
