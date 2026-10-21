SMART AUDIO LAB — NEXT-LEVEL IMPLEMENTATION
============================================

FEATURES
--------
- High-contrast animated dashboard introduction and signal-engine status.
- Three input modes: Simulation, Live laptop microphone, and ESP32 USB serial.
- Input and processed waveform charts.
- Input and processed FFT spectrum charts.
- ML demonstration prediction and feature display.
- Input and processed audio playback players.
- Analog Electronics + DSP + ML architecture tab.
- Research integrity notes and explicit simulation labels.

INSTALLATION — WINDOWS
----------------------
1. Extract the project ZIP.
2. Open Command Prompt in the project folder.
3. Run:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    streamlit run app.py

Open the localhost URL printed by Streamlit, normally http://localhost:8501.
Stop with Ctrl+C.

LIVE MICROPHONE
---------------
1. Choose "Live laptop microphone" in the sidebar.
2. Set sampling rate and duration (start with 1 second at 16,000 Hz).
3. Click "Capture / Run analysis".
4. Allow microphone access if the operating system asks.
5. Open Signal arena to inspect waveform/FFT and Listen & compare to hear input and processed samples.

This captures a short clip on demand, not continuous streaming. Close apps that exclusively occupy the microphone.
If sounddevice/PortAudio fails, check the OS input-device permission and default microphone. On some systems
PortAudio may require separate installation. Reinstall packages from requirements.txt after fixing the device.

SIMULATION MODE
---------------
Select Simulation mode, choose an environment, set noise and voice level, then click Capture / Run analysis.
Simulation is synthetic and must not be reported as a real microphone or hardware measurement.

ESP32 USB SERIAL MODE
---------------------
Select ESP32 via USB serial, connect the board, choose the COM/serial port and click Capture / Run analysis.
Expected firmware output: one newline-terminated CSV line containing at least 64 ADC integers, e.g.:

    2048,2050,2052,2049,2045,2042,2047,2051,...

This is one-block acquisition, not continuous real-time streaming. Close Arduino Serial Monitor before connecting
from this app. For stable 16 kHz streaming, use framed binary samples or another efficient protocol instead of
a long ASCII CSV line.

HARDWARE SAFETY
---------------
Confirm the ADC limits for the exact ESP32 board. Bias and condition the microphone signal correctly, and keep
the ADC input within its allowed voltage range. Never apply negative or over-range voltage to the ADC. An analog
anti-aliasing filter should be installed before ADC sampling.

DSP AND ML LIMITATIONS
----------------------
The demonstration DSP uses DC removal, Butterworth low-pass filtering and a simple energy gate. A low-pass filter
cannot remove every type of environmental noise, and gating may attenuate quiet speech. The ML model is trained
on synthetic feature vectors; scores are not validated real-world accuracy. Train and evaluate on labelled audio
with a held-out test set before making classification claims.

GITHUB
------
Create a GitHub repository and upload app.py, requirements.txt, README.txt and .gitignore.
Or run:

    git init
    git add app.py requirements.txt README.txt .gitignore
    git commit -m "Add live audio capture and playback"
    git branch -M main
    git remote add origin https://github.com/YOUR-USERNAME/smart-audio-lab.git
    git push -u origin main

Replace YOUR-USERNAME with your GitHub username. Do not upload .venv, passwords, API keys, or private audio.

STREAMLIT CLOUD
---------------
The simulation mode can be deployed to Streamlit Community Cloud. Cloud hosting generally cannot access the
microphone or USB serial port attached to your personal laptop. Use the local app for live microphone and ESP32
testing.

JUPYTER
-------
Use Jupyter for experiments, feature testing and model training. Run the dashboard from a notebook cell with:

    !streamlit run app.py

TROUBLESHOOTING
---------------
- If streamlit is not recognized: python -m streamlit run app.py
- If dependencies are missing: pip install -r requirements.txt
- If microphone access fails: check OS privacy settings and selected input device.
- If no serial port appears: check USB data cable, board drivers and device manager.
- If old styling appears: stop Streamlit, restart it and press Ctrl+F5 in the browser.
