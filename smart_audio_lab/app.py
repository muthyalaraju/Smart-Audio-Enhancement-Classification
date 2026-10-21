import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except Exception:
    SOUNDDEVICE_OK = False

try:
    import serial
    from serial.tools import list_ports
    SERIAL_OK = True
except Exception:
    SERIAL_OK = False

st.set_page_config(page_title="Smart Audio Lab", page_icon="🎧", layout="wide")

# ---------- High-contrast animated styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
.stApp {
  background: radial-gradient(circle at 10% 5%,#e1f7ff 0,transparent 25%),
              radial-gradient(circle at 95% 15%,#efe5ff 0,transparent 24%),
              linear-gradient(135deg,#f6f9ff,#edf5fb 58%,#f7f2ff);
  color:#18324d;
  font-family:Inter,sans-serif;
}
.block-container {max-width:1550px;padding-top:1.25rem;padding-bottom:2.4rem}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#e9f2ff,#f9fbff)!important;border-right:1px solid #cbdcf0}
[data-testid="stSidebar"] * {color:#173451!important}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] label p,[data-testid="stWidgetLabel"] p {color:#173451!important;font-weight:650!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div,
[data-testid="stSidebar"] [data-baseweb="input"]>div,
[data-testid="stSidebar"] [data-testid="stNumberInput"] button,
[data-testid="stSidebar"] input {background:#fff!important;color:#173451!important;border:1px solid #bfd2e7!important;border-radius:12px!important}
[data-testid="stSidebar"] [data-baseweb="select"] *,[data-testid="stSidebar"] input {-webkit-text-fill-color:#173451!important;color:#173451!important}
[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"],[role="option"] {background:#fff!important;color:#173451!important}
.hero {position:relative;overflow:hidden;padding:1.8rem;border-radius:25px;color:white!important;
background:linear-gradient(110deg,#10223e,#1f6087 56%,#6652a5);box-shadow:0 16px 42px #19325530;margin-bottom:1rem;
animation:rise .75s ease both}
.hero:before {content:"";position:absolute;inset:0;background:linear-gradient(115deg,transparent 20%,#ffffff20 48%,transparent 65%);
transform:translateX(-120%);animation:shine 5s ease-in-out infinite}
.hero:after {content:"";position:absolute;width:220px;height:220px;right:-55px;top:-100px;border:1px solid #ffffff55;border-radius:50%;
box-shadow:0 0 0 26px #ffffff0a,0 0 0 54px #ffffff08;animation:float 6s ease-in-out infinite}
.hero h1,.hero p,.hero span {position:relative;z-index:1;color:white!important}
.hero h1 {font-size:clamp(1.7rem,3vw,2.5rem);margin:0 0 .4rem}
.hero p {color:#e8f5ff!important;margin:0}
.pill {display:inline-block;border:1px solid #ffffff55;background:#ffffff1c;border-radius:999px;padding:.32rem .75rem;margin:.8rem .3rem 0 0;font-size:.78rem}
div[data-testid="stMetric"] {background:#ffffffef;border:1px solid #d9e6f4;border-radius:17px;padding:.9rem;box-shadow:0 7px 20px #24476b0e;animation:rise .5s ease both;transition:transform .2s,box-shadow .2s}
div[data-testid="stMetric"]:hover {transform:translateY(-4px);box-shadow:0 12px 26px #24476b1c}
[data-testid="stMetricLabel"] {color:#536b83!important;font-weight:700!important}
[data-testid="stMetricValue"] {color:#142c47!important;font-weight:850!important}
.stTabs [data-baseweb="tab"] {color:#42566d!important;font-weight:750;transition:all .2s}
.stTabs [data-baseweb="tab"]:hover {background:#e1efff!important;color:#155f91!important}
.stTabs [aria-selected="true"] {color:#155f91!important;background:#eaf4ff!important}
.stButton button {border-radius:13px;font-weight:800;min-height:2.7rem;transition:transform .18s,box-shadow .18s}
.stButton button:hover {transform:translateY(-2px);box-shadow:0 8px 20px #204a7025}
div[data-testid="stPlotlyChart"] {background:#ffffffdf;border:1px solid #dce7f4;border-radius:17px;padding:.2rem;box-shadow:0 8px 24px #24476b0d;animation:chart .6s ease both;transition:transform .2s}
div[data-testid="stPlotlyChart"]:hover {transform:translateY(-3px)}
.status {background:#ffffffed;border:1px solid #d8e5f3;border-radius:16px;padding:.85rem 1rem;box-shadow:0 6px 18px #24476b0b;min-height:76px;color:#203b57!important}
.status * {color:#203b57!important}
.card {background:linear-gradient(135deg,#fff,#f5faff);border:1px solid #dce7f4;border-radius:17px;padding:1rem;box-shadow:0 7px 22px #24476b0c;animation:rise .55s ease both}
@keyframes rise {from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes chart {from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes float {0%,100%{transform:translateY(0)}50%{transform:translateY(16px) rotate(5deg)}}
@keyframes shine {0%,65%{transform:translateX(-120%)}90%,100%{transform:translateX(120%)}}
@keyframes pulse {0%{box-shadow:0 0 0 0 #13b89a70}70%{box-shadow:0 0 0 9px #13b89a00}100%{box-shadow:0 0 0 0 #13b89a00}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;transition-duration:.01ms!important}}
</style>
""", unsafe_allow_html=True)

if "intro_seen" not in st.session_state:
    splash=st.empty()
    splash.markdown("""
    <div style="position:fixed;z-index:999999;inset:0;background:linear-gradient(125deg,#10223e,#1f6087,#6652a5);display:flex;align-items:center;justify-content:center;text-align:center;color:#fff">
    <div style="padding:2rem;animation:intro .9s ease both">
      <div style="font-size:3.5rem">🎧</div><div style="font-size:2.1rem;font-weight:900;color:white">SMART AUDIO LAB</div>
      <div style="margin-top:.5rem;color:#e5f3ff">ANALOG ELECTRONICS · DSP · MACHINE LEARNING</div>
      <div style="height:5px;width:210px;background:#ffffff44;border-radius:10px;margin:1.5rem auto;overflow:hidden"><div style="height:100%;background:#8df0d1;animation:load 1.2s ease both"></div></div>
      <div style="color:#d8eaff">Booting signal engine…</div>
    </div></div><style>@keyframes intro{from{opacity:0;transform:translateY(15px)}to{opacity:1;transform:translateY(0)}}@keyframes load{from{width:0}to{width:100%}}</style>
    """,unsafe_allow_html=True)
    time.sleep(1.25)
    splash.empty()
    st.session_state.intro_seen=True

st.markdown("""
<div class="hero">
<h1>🎧 Smart Audio Lab</h1>
<p>Capture sound · inspect the spectrum · enhance the signal · classify the environment</p>
<span class="pill">ANALOG FRONT-END</span><span class="pill">LIVE AUDIO</span><span class="pill">DSP FILTERING</span><span class="pill">ML PREDICTION</span>
</div>""",unsafe_allow_html=True)

@st.cache_resource
def train_demo_model():
    rng=np.random.default_rng(23)
    centers={
        "Speech":[.15,.22,.40,.37,.45],
        "Fan / AC":[.08,.10,.20,.16,.30],
        "Traffic":[.20,.16,.36,.60,.38],
        "Crowd / Chatter":[.17,.36,.47,.53,.58],
        "Quiet":[.01,.04,.08,.06,.07],
    }
    X=[];y=[]
    for label,c in centers.items():
        for _ in range(180):
            X.append(np.maximum(0,rng.normal(c,[.04,.09,.12,.13,.13])))
            y.append(label)
    model=make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=120,max_depth=9,random_state=23,class_weight="balanced"))
    model.fit(np.array(X),np.array(y))
    return model
model=train_demo_model()

def synth_signal(kind,seconds,fs,noise,voice,seed):
    n=int(seconds*fs);t=np.arange(n)/fs;rng=np.random.default_rng(seed)
    f0=135+20*np.sin(2*np.pi*.8*t)
    phase=2*np.pi*np.cumsum(f0)/fs
    v=(np.sin(phase)+.48*np.sin(2*phase)+.2*np.sin(3*phase))
    v*= (.55+.45*np.sin(2*np.pi*2*t)**2)*voice
    w=rng.normal(0,1,n)
    if kind=="Fan / AC": ns=.7*np.sin(2*np.pi*115*t)+.25*np.sin(2*np.pi*230*t)+.45*w
    elif kind=="Traffic":
        slow=np.interp(np.linspace(0,1,n),np.linspace(0,1,30),rng.normal(0,1,30))
        ns=.8*slow+.55*w+.15*np.sin(2*np.pi*70*t)
    elif kind=="Crowd / Chatter":
        ns=.45*w
        for f,a in [(190,.23),(280,.18),(370,.14),(510,.1)]: ns+=a*np.sin(2*np.pi*f*t+rng.uniform(0,6.28))
    elif kind=="Quiet": ns=.12*w
    else: ns=.35*w
    x=v+noise*ns
    return (x/(np.max(np.abs(x))+1e-9)*.88).astype(np.float32)

def features(x,fs):
    x=np.asarray(x,float)
    if len(x)<2:return np.zeros(5)
    rms=np.sqrt(np.mean(x*x));zcr=np.mean(np.diff(np.signbit(x))!=0)
    mag=np.abs(np.fft.rfft(x*np.hanning(len(x))))+1e-12;freq=np.fft.rfftfreq(len(x),1/fs)
    centroid=np.sum(freq*mag)/np.sum(mag)
    roll=freq[min(np.searchsorted(np.cumsum(mag),.85*np.sum(mag)),len(freq)-1)]
    low=np.sum(mag[freq<500])/np.sum(mag)
    return np.array([rms,zcr,min(centroid/4000,1),min(roll/8000,1),low])

def process_audio(x,fs,cutoff,enhance_strength):
    x=np.asarray(x,dtype=np.float32)
    x=x-np.mean(x)
    cutoff=float(np.clip(cutoff,100,fs*.45))
    try: filtered=sosfiltfilt(butter(4,cutoff,btype="lowpass",fs=fs,output="sos"),x)
    except Exception: filtered=x.copy()
    # A simple gate for demonstration; it can attenuate quiet speech.
    threshold=np.quantile(np.abs(filtered),.18)*(.2+enhance_strength)
    gate=np.clip((np.abs(filtered)-threshold)/(threshold+1e-6),0,1)
    y=filtered*(.35+.65*gate)
    peak=np.max(np.abs(y))+1e-9
    return (y*min(1,.98/peak)).astype(np.float32)

def plot_wave(x,fs,title,color):
    fig=go.Figure(go.Scatter(x=np.arange(len(x))/fs,y=x,mode="lines",line=dict(color=color,width=1.6),fill="tozeroy",fillcolor="rgba(40,120,180,.08)"))
    fig.update_layout(title=title,height=300,template="plotly_white",margin=dict(l=12,r=12,t=48,b=15),xaxis_title="Time (s)",yaxis_title="Amplitude",transition_duration=500,hovermode="x unified")
    return fig

def plot_fft(x,fs,title,color):
    mag=np.abs(np.fft.rfft(x*np.hanning(len(x))))/max(1,len(x));freq=np.fft.rfftfreq(len(x),1/fs)
    fig=go.Figure(go.Scatter(x=freq,y=mag,mode="lines",line=dict(color=color,width=1.8),fill="tozeroy",fillcolor="rgba(40,120,180,.08)"))
    fig.update_layout(title=title,height=300,template="plotly_white",margin=dict(l=12,r=12,t=48,b=15),xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(range=[0,min(8000,fs/2)]),transition_duration=500,hovermode="x unified")
    return fig

# Sidebar
st.sidebar.markdown("## 🎛️ Audio control room")
source=st.sidebar.radio("Input source",["Simulation mode","Live laptop microphone","ESP32 via USB serial"])
fs=st.sidebar.selectbox("Sampling rate (Hz)",[8000,16000,22050],index=1)
seconds=st.sidebar.select_slider("Capture / analysis duration (s)",[0.5,1.0,2.0,3.0],value=1.0)
environment=st.sidebar.selectbox("Simulation environment",["Speech","Fan / AC","Traffic","Crowd / Chatter","Quiet"],index=2)
noise=st.sidebar.slider("Environmental noise",0.0,1.5,.8,.05)
voice=st.sidebar.slider("Desired voice level",.1,1.0,.65,.05)
cutoff=st.sidebar.slider("DSP low-pass cutoff (Hz)",300,min(7500,int(fs*.45)),min(3400,int(fs*.45)//100*100),100)
strength=st.sidebar.slider("Enhancement strength",0.0,1.0,.6,.05)
st.sidebar.caption("Tip: use headphones to avoid speaker feedback when testing a microphone.")
capture=st.sidebar.button("🚀 Capture / Run analysis",type="primary",use_container_width=True)
if "runs" not in st.session_state: st.session_state.runs=0
if capture: st.session_state.runs+=1

raw=None; source_label="SIMULATION"; source_note="Synthetic signal generated for demonstration."
if source=="Simulation mode":
    raw=synth_signal(environment,seconds,fs,noise,voice,42+st.session_state.runs)
elif source=="Live laptop microphone":
    if not SOUNDDEVICE_OK:
        st.error("Live microphone needs sounddevice and PortAudio. Install requirements, then restart the app.")
    elif capture:
        try:
            with st.spinner("🎙️ Capturing live microphone audio…"):
                recording=sd.rec(int(seconds*fs),samplerate=fs,channels=1,dtype="float32")
                sd.wait()
            raw=recording[:,0]
            source_label="LIVE MICROPHONE"
            source_note=f"Captured from the laptop microphone at {fs:,} Hz."
        except Exception as e:
            st.error(f"Microphone capture failed: {e}. Check OS microphone permission and input device.")
    else:
        st.info("Choose Live laptop microphone, then click Capture / Run analysis. A short live clip will be captured each time.")
elif source=="ESP32 via USB serial":
    if not SERIAL_OK:
        st.error("Hardware input needs pyserial. Install requirements.")
    elif capture:
        ports=[p.device for p in list_ports.comports()]
        if not ports:
            st.error("No serial port detected. Connect the ESP32 or choose Simulation mode.")
        else:
            port=st.sidebar.selectbox("ESP32 serial port",ports)
            baud=st.sidebar.selectbox("Serial baud rate",[115200,230400,460800,921600],index=2)
            st.sidebar.caption("Firmware should send one newline-terminated CSV line of ADC values per block.")
            try:
                with st.spinner("Reading one ADC block from ESP32…"):
                    with serial.Serial(port,baud,timeout=3) as ser:
                        ser.reset_input_buffer()
                        line=ser.readline().decode("utf-8",errors="ignore").strip()
                vals=np.fromstring(line,sep=",")
                if len(vals)<64: raise ValueError("Received fewer than 64 samples. Check firmware and baud rate.")
                raw=((vals-np.mean(vals))/(np.max(np.abs(vals-np.mean(vals)))+1e-9)).astype(np.float32)
                source_label=f"ESP32 HARDWARE · {port}"
                source_note="One serial ADC block received. This is block capture, not continuous streaming."
            except Exception as e: st.error(f"Serial read failed: {e}")

if raw is None:
    raw=np.zeros(max(64,int(fs*seconds)),dtype=np.float32)
    waiting=source!="Simulation mode"
else: waiting=False

processed=process_audio(raw,fs,cutoff,strength)
feat=features(raw,fs)
prediction=model.predict(feat.reshape(1,-1))[0]
probs=model.predict_proba(feat.reshape(1,-1))[0]
rms_in=float(np.sqrt(np.mean(raw**2)));rms_out=float(np.sqrt(np.mean(processed**2)))
freq=np.fft.rfftfreq(len(raw),1/fs);mag=np.abs(np.fft.rfft((raw-np.mean(raw))*np.hanning(len(raw))))
dominant=float(freq[1+np.argmax(mag[1:])]) if len(mag)>1 else 0
change=20*np.log10((rms_out+1e-9)/(rms_in+1e-9))

# HUD status
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin:.4rem 0 1rem;padding:.8rem 1rem;background:linear-gradient(90deg,#e4f8f3,#edf4ff);border:1px solid #cfe6ec;border-radius:15px;color:#1d4860;font-weight:800">
<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#13b89a;animation:pulse 1.7s infinite"></span>
SIGNAL ENGINE READY <span style="font-weight:500;color:#547187">· {source_label} · {source_note}</span>
</div>""",unsafe_allow_html=True)
if source=="Simulation mode":
    st.info("Simulation is active: these are synthetic signals, not microphone recordings or physical hardware readings.")
elif waiting:
    st.warning("Waiting for capture. Click **Capture / Run analysis** to acquire audio from the selected source.")

st.markdown("### ⚡ Live performance snapshot")
metrics=st.columns(5)
for col,label,val in zip(metrics,["Input RMS","Output RMS","Dominant frequency","Samples captured","RMS level change"],
                         [f"{rms_in:.3f}",f"{rms_out:.3f}",f"{dominant:.0f} Hz",f"{len(raw):,}",f"{change:+.1f} dB"]):
    col.metric(label,val)

tab1,tab2,tab3,tab4,tab5=st.tabs(["🎮 Signal arena","🧠 ML prediction","🔊 Listen & compare","🧩 Three-subject system","📘 Research / limitations"])
with tab1:
    st.markdown("### Signal arena · Before and after DSP")
    c1,c2=st.columns(2)
    with c1:
        st.plotly_chart(plot_wave(raw,fs,"01 · INPUT WAVEFORM","#2878b5"),use_container_width=True)
        st.plotly_chart(plot_fft(raw,fs,"02 · INPUT FREQUENCY SPECTRUM","#7c59c9"),use_container_width=True)
    with c2:
        st.plotly_chart(plot_wave(processed,fs,"03 · PROCESSED WAVEFORM","#119b84"),use_container_width=True)
        st.plotly_chart(plot_fft(processed,fs,"04 · PROCESSED FREQUENCY SPECTRUM","#ef7b37"),use_container_width=True)
    st.caption("Graph transitions animate when a new analysis is submitted. A visual change is not, by itself, proof of perceptual improvement.")
with tab2:
    left,right=st.columns([.8,1.2])
    with left:
        st.metric("Predicted environment",prediction)
        st.metric("Top demonstration score",f"{np.max(probs)*100:.1f}%")
        st.warning("Demonstration model only: trained on synthetic feature examples. Scores are not validated real-world accuracy.")
    with right:
        score_df=pd.DataFrame({"Environment":model.classes_,"Score":probs})
        fig=go.Figure(go.Bar(x=score_df.Score,y=score_df.Environment,orientation="h",
            marker_color=["#16a085" if label==prediction else "#9aacc1" for label in score_df.Environment],
            text=[f"{v*100:.1f}%" for v in score_df.Score],textposition="outside"))
        fig.update_layout(title="Environment prediction scores",height=340,template="plotly_white",xaxis=dict(range=[0,1.15],title="Model score"),transition_duration=600)
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Extracted DSP features")
    st.dataframe(pd.DataFrame({"Feature":["RMS energy","Zero-crossing rate","Spectral centroid (scaled)","Spectral roll-off (scaled)","Low-band energy ratio"],"Value":feat}),use_container_width=True,hide_index=True)
with tab3:
    st.markdown("### 🔊 Listen to input and processed audio")
    st.write("Use these players to compare the captured input with the current DSP output. Live microphone mode captures a short clip when you click the button.")
    a,b=st.columns(2)
    with a:
        st.markdown("#### Original / input audio")
        st.audio(raw, sample_rate=fs)
    with b:
        st.markdown("#### DSP processed audio")
        st.audio(processed, sample_rate=fs)
    st.warning("The demonstration low-pass filter and simple energy gate may also remove useful speech. Listen critically and adjust the cutoff/strength.")
with tab4:
    st.markdown("""
    <div class="card">
    <h3 style="color:#173451">01 · Analog Electronics</h3>
    <p>Electret microphone → bias network → LM358 pre-amplifier → analog anti-aliasing filter → ESP32 ADC.</p>
    <h3 style="color:#173451">02 · Digital Signal Processing</h3>
    <p>Sampling → DC removal → Hamming-window FFT → feature extraction → Butterworth low-pass filtering and simple energy gate.</p>
    <h3 style="color:#173451">03 · Machine Learning</h3>
    <p>Signal features → environment prediction → future context-aware selection of an enhancement profile.</p>
    <h3 style="color:#173451">Output</h3>
    <p>The dashboard now provides browser audio players for input and processed samples. For hardware, it can read a sample block sent by compatible ESP32 firmware over USB serial.</p>
    </div>""",unsafe_allow_html=True)
    st.code("Microphone → analog preamp/filter → ADC → Python DSP/FFT → feature extraction → ML → enhancement → audio playback",language="text")
with tab5:
    st.markdown("""
    - Simulation signals are generated in software; they are not real measurements.
    - Live microphone mode captures a short clip on demand, not a continuously streaming microphone.
    - ESP32 mode reads one newline-terminated CSV block of ADC values; continuous streaming requires a separate framed streaming implementation.
    - The ML Random Forest is trained on synthetic feature vectors and is not a validated real-world classifier.
    - The low-pass filter does not remove every kind of noise; the energy gate may attenuate quiet speech.
    - For research claims, evaluate SNR, intelligibility/audio quality, end-to-end latency, held-out classification accuracy, precision, recall, F1 and a confusion matrix.
    """)

st.markdown("---")
st.markdown("<p style='color:#63768b;font-size:.88rem'>SMART AUDIO LAB · Educational prototype · Distinguishes simulation from live and hardware inputs</p>",unsafe_allow_html=True)
