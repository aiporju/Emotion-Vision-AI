import streamlit as st
from PIL import Image
from deepface import DeepFace
import tempfile
import time


# -------------------------
# PAGE SETTINGS
# -------------------------

st.set_page_config(
    page_title="Emotion Vision AI",
    page_icon="🧠",
    layout="wide"
)



# -------------------------
# EMOTION DATA
# -------------------------

EMOJIS = {

    "happy":"😄",
    "sad":"😢",
    "angry":"😠",
    "surprise":"😲",
    "fear":"😨",
    "neutral":"😐",
    "disgust":"🤢"

}



QUOTES = {

"happy":
"🌟 You look happy! Keep spreading positivity.",

"sad":
"💙 Take a short break and relax your mind.",

"angry":
"🧘 Try deep breathing and stay calm.",

"surprise":
"✨ Something interesting caught your attention!",

"fear":
"🌱 Stay strong. Every challenge helps you grow.",

"neutral":
"📚 You seem calm and focused.",

"disgust":
"🌿 Stay positive and balanced."

}



# -------------------------
# HEADER
# -------------------------

st.markdown(
"""
<h1 style='text-align:center'>
🧠🌈 Emotion Vision AI
</h1>

<h3 style='text-align:center'>
Computer Vision Based Emotion Detection System
</h3>

""",
unsafe_allow_html=True
)



st.divider()



# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.title("⚙️ AI Information")


st.sidebar.info(
"""
Technology Used:

👁 Computer Vision

🤖 Deep Learning

🙂 Facial Emotion Recognition


Flow:

Image
↓
Face Detection
↓
Emotion Analysis
↓
Prediction
"""
)



choice = st.sidebar.radio(

"Choose Input",

[
"Upload Image",
"Camera"
]

)



# -------------------------
# IMAGE INPUT
# -------------------------

image = None



if choice == "Upload Image":


    uploaded = st.file_uploader(

        "📂 Upload Face Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )


    if uploaded:

        image = Image.open(uploaded)



else:


    camera = st.camera_input(

        "📸 Capture Image"

    )


    if camera:

        image = Image.open(camera)




# -------------------------
# ANALYSIS
# -------------------------

if image:


    # resize for speed

    image = image.resize(
        (500,500)
    )


    col1,col2 = st.columns(2)



    with col1:


        st.subheader(
            "📷 Input Image"
        )


        st.image(

            image,

            use_container_width=True

        )




    with col2:


        st.subheader(
            "🤖 AI Result"
        )



        button = st.button(

            "🔍 Detect Emotion",

            use_container_width=True

        )



        if button:


            with st.spinner(
                "AI analysing facial features..."
            ):


                time.sleep(1)



                with tempfile.NamedTemporaryFile(

                    suffix=".jpg",

                    delete=False

                ) as temp:


                    image.save(
                        temp.name
                    )



                    result = DeepFace.analyze(

                        img_path=temp.name,

                        actions=[
                            "emotion"
                        ],


                        # Faster detector

                        detector_backend="opencv",


                        align=True,


                        enforce_detection=False

                    )




            emotion = result[0]["dominant_emotion"]



            confidence = round(

                result[0]["emotion"][emotion],

                2

            )




            st.success(
                "✅ Analysis Complete"
            )



            st.markdown(

            f"""

            <div style='

            background:#eeeeee;

            padding:20px;

            border-radius:15px;

            text-align:center;

            '>


            <h1>

            {EMOJIS[emotion]}

            </h1>


            <h2>

            {emotion.upper()}

            </h2>


            </div>

            """,

            unsafe_allow_html=True

            )




            st.metric(

                "🎯 Confidence",

                f"{confidence}%"

            )



            st.progress(

                int(confidence)

            )




            st.info(

                QUOTES[emotion]

            )



            st.subheader(

                "📊 Emotion Probability"

            )



            st.bar_chart(

                result[0]["emotion"]

            )





# -------------------------
# ABOUT AI
# -------------------------

st.divider()



st.subheader(

"🧠 How Computer Vision Works"

)



st.write(

"""

1️⃣ Image is provided to AI

2️⃣ Face is detected using computer vision

3️⃣ Facial patterns are analysed

4️⃣ Deep learning model predicts emotions

5️⃣ Highest probability emotion is displayed

"""

)



st.warning(

"""
⚠️ AI Note:
Emotion prediction depends on lighting,
camera angle and facial expression.
"""

)



# -------------------------
# FOOTER
# -------------------------

st.divider()


st.caption(

"🏆 Class 12 AI Project | Human Emotion Sensor"

)
