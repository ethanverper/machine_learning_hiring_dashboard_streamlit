import streamlit as st

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import pandas as pd
import scikitplot as skplt
from joblib import dump, load
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeClassifier

categories = [
    'alt.atheism',
    'comp.graphics',
    'comp.sys.mac.hardware',
    'rec.autos',
    'sci.space',
    'sci.med',
    'sci.electronics',
    'talk.politics.guns',
    'soc.religion.christian',
]

import streamlit_option_menu
from streamlit_option_menu import option_menu

import streamlit as st

# Configuración de autenticación simple
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]["password"]:
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False
            st.error("❌ Incorrect password. Access denied.")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        with st.form("Login"):
            st.text_input("Enter Dashboard Password", type="password", key="password")
            submitted = st.form_submit_button("🔐 Login")
            if submitted:
                password_entered()
        return False
    else:
        return True

if not check_password():
    st.stop()

with st.sidebar:
  selected = option_menu(
    menu_title = "Main Menu",
    options = ["Home", "Econometric Model", "Data Science Model", "Models Comparison", "Extra Example"],
    icons = ["house", "graph-up", "cpu", "bar-chart-line", "code-square"],
    menu_icon = "cast",
    default_index = 0,

  )
import openai
import streamlit as st
import seaborn as sns

client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

def ai_insight_block(context: str, label_prefix="🧠"):
    with st.expander(f"{label_prefix} Ask the AI Assistant for Insights"):
        user_question = st.text_area("What would you like to ask about this section?", key=f"{label_prefix}_q")
        
        if st.button("🔍 Analyze with AI", key=f"{label_prefix}_btn"):
            if user_question.strip():
                with st.spinner("Thinking..."):
                    prompt = f"""
You are a helpful machine learning assistant. The user is analyzing the following model output:

{context}

They asked:
{user_question}

Provide an accurate, concise, and professional explanation.
"""
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant providing insights into ML model outputs."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    answer = response.choices[0].message.content
                    st.markdown("### 💬 Assistant Answer")
                    st.info(answer)
            else:
                st.warning("❗ Please enter a question to get insights.")
                
@st.cache_data
def load_recruitment_data():
    df = pd.read_csv("recruitment_data.csv")

    X = df.drop(columns=["HiringDecision"])
    y = df["HiringDecision"]

    return X, y
  
@st.cache_data
def load_and_vectorize_dataset():
    ## Load Dataset
    news_groups = datasets.fetch_20newsgroups(categories=categories)
    X_train, X_test, Y_train, Y_test = train_test_split(news_groups.data, news_groups.target, train_size=0.8, random_state=123)

    ## Vectorize Data
    vectorizer = CountVectorizer(max_features=50_000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, Y_train, Y_test, vectorizer

X_train_vec, X_test_vec, Y_train, Y_test, vectorizer = load_and_vectorize_dataset()

def train_model(n_estimators, max_depth, max_features, bootstrap):
    rf_classif = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, bootstrap=bootstrap)
    rf_classif.fit(X_train_vec, Y_train)
    return rf_classif

def train_decision_tree(max_depth, min_samples_split, criterion):
    from sklearn.model_selection import train_test_split

    X, y = load_recruitment_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

# Carga del dataset de contratación si no está en sesión
if "df_hiring" not in st.session_state:
    df_hiring = pd.read_csv("recruitment_data.csv")  # o el nombre correcto que uses
    st.session_state["df_hiring"] = df_hiring


if selected == "Home":
    
    st.title("📁 Project Overview – Hiring Decisions Dataset")

    st.markdown("""
    This project analyzes a recruitment dataset to explore how candidate attributes influence hiring decisions.

    The application aims to **predict the hiring outcome** based on several features such as education, skills, personality, interview score, and more. It compares **traditional econometric modeling** (Logistic Regression) with **modern classification models** (Decision Tree).

    ---
    ### 🧭 Application Structure
    - 🧠 **Data Introduction & EDA** (this page)
    - ⚙️ **Model Training & Evaluation** (Econometric / Data Science tabs)
    - 📊 **Model Comparison** (side-by-side metrics & visualizations)
    - 📋 **Prediction Reports** (final output and summary)

    ---
    ### 📂 Dataset Overview
    The dataset contains **candidates' information** relevant to job applications, such as:
    - Age, Gender, and Education Level
    - Experience (Years), Previous Companies
    - Distance to Company
    - Scores: Interview, Skill, Personality
    - Recruitment Strategy used
    - Final Hiring Decision (0 = Rejected, 1 = Hired)
    """)

    # Load dataset from session
    df = st.session_state["df_hiring"]

    st.markdown("### 📑 Sample Records")
    st.dataframe(df.head(10))

    st.markdown("---")
    st.markdown("### 📌 Feature Distribution")

    # Education Level
    st.subheader("🎓 Education Level")
    edu_counts = df["EducationLevel"].value_counts().sort_index()
    edu_labels = {
        1: "High School",
        2: "Bachelor",
        3: "Master",
        4: "PhD"
    }
    edu_counts.index = edu_counts.index.map(edu_labels)
    fig_edu, ax_edu = plt.subplots()
    ax_edu.bar(edu_counts.index, edu_counts.values, color="#64b5f6")
    ax_edu.set_title("Distribution of Education Levels")
    ax_edu.set_ylabel("Number of Candidates")
    st.pyplot(fig_edu)
    st.info("Most candidates hold Bachelor's or Master's degrees. A small number have only high school education or a PhD.")

    # Experience
    st.subheader("💼 Years of Experience")
    fig_exp, ax_exp = plt.subplots()
    sns.histplot(df["ExperienceYears"], kde=True, bins=15, color="#4db6ac", ax=ax_exp)
    ax_exp.set_title("Years of Professional Experience")
    ax_exp.set_xlabel("Experience (Years)")
    st.pyplot(fig_exp)
    st.info("Experience varies widely, with a concentration around 5–10 years.")

    # Interview Score
    st.subheader("🗣️ Interview Score")
    fig_interview, ax_interview = plt.subplots()
    sns.histplot(df["InterviewScore"], kde=True, bins=15, color="#ffb74d", ax=ax_interview)
    ax_interview.set_title("Distribution of Interview Scores")
    ax_interview.set_xlabel("Interview Score")
    st.pyplot(fig_interview)
    st.info("Interview scores are moderately skewed, suggesting variability in interview performance.")

    # Personality Score
    st.subheader("🎭 Personality Score")
    fig_pers, ax_pers = plt.subplots()
    sns.boxplot(df["PersonalityScore"], color="#ba68c8", ax=ax_pers)
    ax_pers.set_title("Boxplot of Personality Scores")
    st.pyplot(fig_pers)
    st.info("Most candidates score above 60 in personality, though outliers exist.")

    # ========================
    # HIRING CATEGORIES BLOCK
    # ========================
    st.markdown("## 🗂️ Recruitment Strategy Categories")

    st.markdown("""
    This project uses **3 key categories** from the hiring dataset to represent different recruitment strategies.  
    These groups reflect diverse methods: from internal HR processes to external partners and data-driven AI selections.
    """)

    # Temáticas personalizadas para el dataset de hiring
    recruitment_groups = {
        "Human Decision-Making": ["1"],
        "AI & Automated Systems": ["2"],
        "Outsourced / External Hiring": ["3"]
    }

    # Visualización
    import matplotlib.pyplot as plt

    labels = list(recruitment_groups.keys())
    sizes = [df[df["RecruitmentStrategy"] == int(c[0])].shape[0] for c in recruitment_groups.values()]
    colors = plt.get_cmap("Set3").colors[:len(labels)]

    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90, colors=colors)
    ax_pie.axis("equal")
    st.pyplot(fig_pie)
    st.caption("📊 Proportion of Candidates by Recruitment Strategy")

    # Mostrar descripción de cada categoría
    st.markdown("### 🔍 Strategy Descriptions")
    for theme, codes in recruitment_groups.items():
        st.markdown(f"#### {theme}")
        if codes[0] == "1":
            st.caption("👥 Candidates selected manually by recruiters based on traditional screening.")
        elif codes[0] == "2":
            st.caption("🤖 Candidates filtered through automated systems or AI-based scoring tools.")
        elif codes[0] == "3":
            st.caption("🌐 Candidates referred or sourced from third-party agencies or platforms.")

    # Nota final
    st.info("These 3 categories are used to evaluate how hiring strategy influences model prediction and generalization across human, automated, and external decision-making channels.")

    # Hiring Decision
    st.subheader("✅ Hiring Decision Distribution")
    decision_counts = df["HiringDecision"].value_counts().sort_index()
    decision_labels = {0: "Rejected", 1: "Hired"}
    decision_counts.index = decision_counts.index.map(decision_labels)
    fig_dec, ax_dec = plt.subplots()
    ax_dec.bar(decision_counts.index, decision_counts.values, color=["#ef5350", "#66bb6a"])
    ax_dec.set_title("Hiring Outcomes")
    st.pyplot(fig_dec)
    st.info("Roughly balanced hiring outcomes, indicating non-trivial prediction challenge.")
        
    st.title("📚 Project Overview – 20 Newsgroups Classification")
    st.markdown("""
    This project focuses on classifying text documents from the **20 Newsgroups** dataset using machine learning techniques.
    
    The goal is to develop and evaluate a Random Forest classifier that can accurately assign each document to one of several predefined categories such as technology, politics, religion, science, or sports.
    
    The application is divided into the following key components:
    - **🔍 Dataset Introduction & EDA (this page)**
    - **⚙️ Model Training (Extra Example Menu)**
    - **📈 Evaluation Metrics**
    - **📊 Predictions & Results**
    """)

    # Mostrar dataset y categoría seleccionada
    st.markdown("## 📂 Dataset Overview")

    from sklearn.datasets import fetch_20newsgroups
    import pandas as pd
    import matplotlib.pyplot as plt

    # Cargar los datos
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
    df = pd.DataFrame({
        "text": newsgroups.data,
        "target": newsgroups.target
    })
    df["category"] = df["target"].apply(lambda x: newsgroups.target_names[x])

        # Sección: Presentación de categorías
    st.markdown("## 🗂️ Categories in Use")

    st.markdown("""
    This project uses **12 curated categories** from the 20 Newsgroups dataset.  
    They span diverse topics across **technology, science, society, and lifestyle** to evaluate the model's ability to generalize over real-world, multi-topic text.
    """)

    # Clasificación por tema
    thematic_groups = {
        "Technology": [
            "comp.graphics", "comp.sys.mac.hardware"
        ],
        "Science & Health": [
            "sci.space", "sci.med", "sci.electronics"
        ],
        "Politics & Society": [
            "talk.politics.guns", "talk.politics.mideast", "soc.religion.christian", "alt.atheism"
        ],
        "Lifestyle & Sports": [
            "rec.autos", "rec.sport.hockey"
        ],
        "Marketplace": [
            "misc.forsale"
        ]
    }

    # Distribución visual por grupo temático
    import matplotlib.pyplot as plt

    labels = list(thematic_groups.keys())
    sizes = [len(v) for v in thematic_groups.values()]
    colors = plt.get_cmap("tab20c").colors[:len(labels)]

    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90, colors=colors)
    ax_pie.axis("equal")
    st.pyplot(fig_pie)
    st.caption("📊 Number of categories per topic area")

    # Mostrar cada categoría en columnas organizadas por tema
    for theme, cats in thematic_groups.items():
        st.markdown(f"#### {theme}")
        cols = st.columns(len(cats))
        for idx, cat in enumerate(cats):
            with cols[idx]:
                if cat == "alt.atheism":
                    st.markdown("🧠 **alt.atheism**")
                    st.caption("Debates on atheism, religious criticism and philosophy.")
                elif cat == "comp.graphics":
                    st.markdown("🖼️ **comp.graphics**")
                    st.caption("3D graphics, rendering, image processing, OpenGL.")
                elif cat == "comp.sys.mac.hardware":
                    st.markdown("💻 **comp.sys.mac.hardware**")
                    st.caption("Mac hardware troubleshooting, upgrades, peripherals.")
                elif cat == "rec.autos":
                    st.markdown("🚗 **rec.autos**")
                    st.caption("Cars, mechanics, brands, and driving discussions.")
                elif cat == "rec.sport.hockey":
                    st.markdown("🏒 **rec.sport.hockey**")
                    st.caption("Hockey leagues, scores, teams, and fan debates.")
                elif cat == "sci.space":
                    st.markdown("🚀 **sci.space**")
                    st.caption("NASA, astronomy, rocket launches, space physics.")
                elif cat == "sci.med":
                    st.markdown("🩺 **sci.med**")
                    st.caption("Medicine, treatments, healthcare and diseases.")
                elif cat == "sci.electronics":
                    st.markdown("🔌 **sci.electronics**")
                    st.caption("Circuits, electronic components, repair guides.")
                elif cat == "talk.politics.guns":
                    st.markdown("🔫 **talk.politics.guns**")
                    st.caption("Gun control, Second Amendment, firearm policy.")
                elif cat == "talk.politics.mideast":
                    st.markdown("🌍 **talk.politics.mideast**")
                    st.caption("Middle Eastern politics, global diplomacy.")
                elif cat == "soc.religion.christian":
                    st.markdown("✝️ **soc.religion.christian**")
                    st.caption("Theology, scripture, spiritual discussion.")
                elif cat == "misc.forsale":
                    st.markdown("💰 **misc.forsale**")
                    st.caption("Classified ads and second-hand items for sale.")

    st.info("These 12 categories were carefully selected to balance thematic diversity and modeling complexity — allowing your classifier to generalize across technical, social, and lifestyle content.")


    # Distribución de clases
    st.markdown("### 📊 Class Distribution")
    class_dist = df["category"].value_counts().sort_values(ascending=False)
    st.bar_chart(class_dist)

    # Distribución de longitud de texto
    st.markdown("### 📏 Text Length Distribution")
    df["text_length"] = df["text"].apply(lambda x: len(x.split()))
    fig_len, ax_len = plt.subplots(figsize=(8, 4))
    df["text_length"].hist(bins=50, ax=ax_len)
    ax_len.set_title("Distribution of Document Length (in words)")
    ax_len.set_xlabel("Word Count")
    ax_len.set_ylabel("Frequency")
    ax_len.set_xlim(0, 2000)
    st.pyplot(fig_len)

    # Ejemplo de documento
    st.markdown("### 🧪 Sample Document")
    idx = st.slider("Select sample index:", min_value=0, max_value=len(df)-1, value=0)
    st.write(f"**Category:** {df.loc[idx, 'category']}")
    st.text(df.loc[idx, "text"][:1000] + "...")

elif selected == "Econometric Model":
    st.title("Econometric Model – Logistic Regression 📊")
    st.markdown("Train and evaluate a classic econometric classifier using logistic regression.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns

    X, y = load_recruitment_data()

    with st.form("logit_form"):
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown("### ⚙️ Model Parameters")
            penalty = st.selectbox("Regularization", ["l2", "none"])
            C = st.slider("Inverse of Regularization Strength (C)", 0.01, 10.0, 1.0)
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
            max_iter = st.slider("Max Iterations", 100, 1000, 200)
            submit_logit = st.form_submit_button("🚀 Train Logistic Model")

        if submit_logit:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            log_model = LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=max_iter,
                random_state=42
            )
            log_model.fit(X_train, y_train)

            st.session_state["log_model"] = log_model
            st.session_state["X_train_log"] = X_train
            st.session_state["X_test_log"] = X_test
            st.session_state["y_train_log"] = y_train
            st.session_state["y_test_log"] = y_test

            y_train_pred = log_model.predict(X_train)
            y_test_pred = log_model.predict(X_test)

            with col2:
                col21, col22 = st.columns(2, gap="medium")
                with col21:
                    st.metric("Test Accuracy", value="{:.2f} %".format(100 * accuracy_score(y_test, y_test_pred)))
                with col22:
                    st.metric("Train Accuracy", value="{:.2f} %".format(100 * accuracy_score(y_train, y_train_pred)))

                st.markdown("### Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4.5))
                sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_title("Confusion Matrix")
                ax_cm.set_xlabel("Predicted label")
                ax_cm.set_ylabel("True label")
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)

            st.success("✅ Logistic Regression model trained successfully!")

    if "log_model" in st.session_state:
        model = st.session_state["log_model"]
        X_train = st.session_state["X_train_log"]
        X_test = st.session_state["X_test_log"]
        y_train = st.session_state["y_train_log"]
        y_test = st.session_state["y_test_log"]
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        st.markdown("## 📊 Model Results Overview")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
            "📋 Classification Report",
            "📈 ROC Curve",
            "📦 Probability Distribution",
            "🧮 Calibration Curve",
            "📊 Logistic Coefficients",
            "📉 Error Analysis by Class",
            "🪜 Lift & Gain Chart",
            "📊 Predicted vs Actual",
            "📈 Precision-Recall Curve",
            "⚖️ Threshold Optimization",
            "📉 Residual Analysis",
            "🔍 Top Misclassified"
        ])

        with tab1:
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose().round(2)

            summary_metrics = df_report.loc[["accuracy", "macro avg", "weighted avg"]]
            class_metrics = df_report.drop(["accuracy", "macro avg", "weighted avg"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Macro F1 Score", f"{summary_metrics.loc['macro avg', 'f1-score']:.2f}")
            with col2:
                st.metric("Overall Accuracy", f"{summary_metrics.loc['accuracy', 'precision']:.2f}")

            st.dataframe(
                class_metrics.style
                .background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])
                .format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"})
            )

            st.markdown("### 📘 Interpretation")
            st.info("""
        - **Precision** is the ratio of correct positive predictions to total predicted positives.
        - **Recall** is the ratio of correct positive predictions to all actual positives.
        - **F1-score** is the harmonic mean of precision and recall.

        Balanced values across classes indicate fair performance. Pay special attention to class imbalance and false positives/negatives.
        """)

        with tab2:
            from sklearn.metrics import roc_curve, auc
            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            auc_score = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_title("ROC Curve")
            ax_roc.set_xlabel("FPR")
            ax_roc.set_ylabel("TPR")
            ax_roc.legend()
            st.pyplot(fig_roc)

            st.markdown("### 📘 Interpretation")
            st.info(f"""
        AUC Score: **{auc_score:.2f}**

        - AUC > 0.90: ✅ Excellent
        - AUC > 0.80: 👍 Very Good
        - AUC > 0.70: 🟡 Acceptable
        - AUC < 0.70: ❌ Poor

        This score summarizes the model's ability to distinguish between classes. The closer to 1.0, the better.
        """)

        with tab3:
            import seaborn as sns
            fig_prob, ax_prob = plt.subplots()
            sns.histplot(y_probs, bins=20, kde=True, ax=ax_prob)
            ax_prob.set_title("Probability Distribution")
            st.pyplot(fig_prob)

            st.markdown("### 📘 Interpretation")
            confident = (y_probs > 0.9).mean()
            uncertain = ((y_probs > 0.4) & (y_probs < 0.6)).mean()
            st.info(f"""
        This plot shows how confident the model is when making predictions.

        - **{confident:.1%}** of predictions were made with high confidence (> 90%).
        - **{uncertain:.1%}** fall in the uncertainty zone (40–60%).

        High uncertainty may suggest borderline cases or areas for further model calibration.
        """)

        with tab4:
            from sklearn.calibration import calibration_curve
            prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10, strategy="uniform")
            fig_cal, ax_cal = plt.subplots()
            ax_cal.plot(prob_pred, prob_true, marker='o', label="Logit Model")
            ax_cal.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_cal.set_title("Calibration Curve")
            ax_cal.set_xlabel("Mean Predicted Probability")
            ax_cal.set_ylabel("Fraction of Positives")
            ax_cal.legend()
            st.pyplot(fig_cal)

            st.markdown("### 📘 Interpretation")
            st.info("""
        This curve shows how well predicted probabilities align with actual outcomes.

        - A perfectly calibrated model would follow the diagonal.
        - Points above the line indicate under-confidence.
        - Points below the line indicate over-confidence.

        Calibration is important in decision-critical domains (e.g., finance, health).
        """)

        with tab5:
            st.markdown("## 🔍 Model Coefficients")
            coef_df = pd.DataFrame({
                "Feature": X_train.columns,
                "Coefficient": model.coef_[0].round(2)
            }).sort_values(by="Coefficient", ascending=False)

            st.dataframe(coef_df.style
                .background_gradient(cmap="PuOr", subset=["Coefficient"])
            )

            st.markdown("### 📘 Interpretation")
            st.info("""
        Model coefficients indicate the **direction** and **magnitude** of influence for each feature:

        - Positive values ⬆️ increase the likelihood of a 'Hired' outcome.
        - Negative values ⬇️ decrease the likelihood.
        - Near-zero values may be less informative.

        The larger the absolute value, the more influential the variable.
        """)
            
        with tab6:
            st.markdown("### 📉 Error Analysis by Class")
            error_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            error_df["Correct"] = (error_df["Actual"] == error_df["Predicted"]).astype(int)
            fig_err, ax_err = plt.subplots()
            sns.countplot(data=error_df, x="Actual", hue="Correct", palette={1: "green", 0: "red"}, ax=ax_err)
            ax_err.set_title("Correct vs Incorrect Predictions by Class")
            st.pyplot(fig_err)
            st.markdown("### 📘 Interpretation")
            st.info("This chart shows model performance per class. High red bars indicate frequent misclassifications.")

        with tab7:
            from sklearn.metrics import roc_auc_score
            import scikitplot as skplt

            st.markdown("### 🪜 Lift Curve & Gain Chart")
            y_proba_full = model.predict_proba(X_test)

            skplt.metrics.plot_lift_curve(y_test, y_proba_full)
            st.pyplot(plt.gcf())

            skplt.metrics.plot_cumulative_gain(y_test, y_proba_full)
            st.pyplot(plt.gcf())

            st.markdown("### 📘 Interpretation")
            st.info("Lift and Gain charts help assess how much better the model performs vs random targeting.")

        with tab8:
            st.markdown("### 📊 Predicted vs Actual (Heatmap)")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Hired", "Hired"], yticklabels=["Not Hired", "Hired"], ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)
            st.markdown("### 📘 Interpretation")
            st.info("This matrix shows how predictions match actual labels. Diagonal cells represent correct predictions.")

        with tab9:
            from sklearn.metrics import precision_recall_curve
            precision, recall, _ = precision_recall_curve(y_test, y_probs)
            fig_pr, ax_pr = plt.subplots()
            ax_pr.plot(recall, precision, color="purple")
            ax_pr.set_title("Precision-Recall Curve")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            st.pyplot(fig_pr)
            st.markdown("### 📘 Interpretation")
            st.info("This curve highlights the trade-off between precision and recall at various thresholds.")

        with tab10:
            st.markdown("### ⚖️ Threshold Optimization")
            thresholds = np.arange(0.0, 1.01, 0.01)
            accuracy_scores = []
            precision_scores = []
            recall_scores = []

            for t in thresholds:
                preds = (y_probs >= t).astype(int)
                accuracy_scores.append(accuracy_score(y_test, preds))
                precision_scores.append(np.sum((preds == 1) & (y_test == 1)) / max(np.sum(preds == 1), 1))
                recall_scores.append(np.sum((preds == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1))

            fig_thresh, ax_thresh = plt.subplots()
            ax_thresh.plot(thresholds, accuracy_scores, label="Accuracy")
            ax_thresh.plot(thresholds, precision_scores, label="Precision")
            ax_thresh.plot(thresholds, recall_scores, label="Recall")
            ax_thresh.set_xlabel("Threshold")
            ax_thresh.set_ylabel("Score")
            ax_thresh.set_title("Threshold Optimization")
            ax_thresh.legend()
            st.pyplot(fig_thresh)
            st.markdown("### 📘 Interpretation")
            st.info("Use this to decide the best threshold depending on your priority: precision, recall or accuracy.")

        with tab11:
            st.markdown("### 📉 Residual Analysis")
            residuals = y_test - y_probs
            fig_resid, ax_resid = plt.subplots()
            sns.histplot(residuals, kde=True, color="gray", ax=ax_resid)
            ax_resid.set_title("Distribution of Residuals")
            st.pyplot(fig_resid)
            st.markdown("### 📘 Interpretation")
            st.info("Residuals close to 0 indicate accurate predictions. A skewed distribution may signal bias.")

        with tab12:
            st.markdown("### 🔍 Top N Misclassified Examples")
            n = st.slider("Select Top N", 5, 50, 10)
            probs = model.predict_proba(X_test)[:, 1]
            errors_df = pd.DataFrame({
                "True": y_test,
                "Predicted": y_pred,
                "Probability": probs
            })
            errors_df = errors_df[errors_df["True"] != errors_df["Predicted"]]
            errors_df["Confidence"] = abs(errors_df["Probability"] - 0.5)
            top_errors = errors_df.sort_values("Confidence").head(n)
            st.dataframe(top_errors.reset_index(drop=True))
            st.markdown("### 📘 Interpretation")
            st.info("These are the most uncertain misclassified samples. Investigate feature values for improvement insights.")

elif selected == "Extra Example":
    
    ## Dashboard
    st.title("Random Forest :green[Experiment] :computer:")
    st.markdown("Try different values of random forest classifier. Select widget values and submit model for training. Various ML metrics will be displayed after training.")

    with st.form("train_model"):
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            n_estimators = st.slider("No of Estimators:", min_value=100, max_value=1000)
            max_depth = st.slider("Max Depth:", min_value=2, max_value=20)
            max_features = st.selectbox("Max Features :", options=["sqrt", "log2", "auto", None, 0.5, 0.75, 5])
            bootstrap = st.checkbox("Bootstrap")
            save_model = st.checkbox("Save Model")

            col_submit1, col_submit2 = st.columns([2, 1])
            with col_submit1:
                submitted = st.form_submit_button("🚀 Train")
            with col_submit2:
                if st.form_submit_button("🔄 Reset View"):
                    st.session_state.clear()
                    st.success("Resetting view...")
                    time.sleep(1)
                    st.experimental_set_query_params(dummy=str(time.time()))  # provoca recarga


        if submitted:
            st.success("✅ Model retrained with selected hyperparameters!")
            rf_classif = train_model(n_estimators, max_depth, max_features, bootstrap)

            if save_model:
                dump(rf_classif, "rf_classif.dat")

            Y_test_preds = rf_classif.predict(X_test_vec)
            Y_train_preds = rf_classif.predict(X_train_vec)
            Y_test_probs = rf_classif.predict_proba(X_test_vec)


            with col2:
                col21, col22 = st.columns(2, gap="medium")
                with col21:
                    st.metric("Test Accuracy", value="{:.2f} %".format(100*accuracy_score(Y_test, Y_test_preds)))
                with col22:
                    st.metric("Train Accuracy", value="{:.2f} %".format(100*accuracy_score(Y_train, Y_train_preds)))

                st.markdown("### Confusion Matrix")
                conf_mat_fig = plt.figure(figsize=(6,6))
                ax1 = conf_mat_fig.add_subplot(111)
                skplt.metrics.plot_confusion_matrix(Y_test, Y_test_preds, ax=ax1)
                st.pyplot(conf_mat_fig, use_container_width=True)

            st.markdown("### Classification Report:")
            # ─── METRIC SUMMARY ─────────────────────────────
            from sklearn.metrics import classification_report
            import pandas as pd

            report_dict = classification_report(Y_test, Y_test_preds, target_names=categories, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose().round(2)

            # Separar métricas globales (accuracy, macro avg, weighted avg)
            summary_metrics = report_df.loc[["accuracy", "macro avg", "weighted avg"]]
            class_metrics = report_df.drop(["accuracy", "macro avg", "weighted avg"])

            # Mostrar KPIs principales
            st.markdown("### 🔢 Model Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Macro F1 Score", f"{summary_metrics.loc['macro avg', 'f1-score']:.2f}")
            with col2:
                st.metric("Overall Accuracy", f"{summary_metrics.loc['accuracy', 'precision']:.2f}")

            # Mostrar tabla estilizada con colores
            st.markdown("### 📋 Classification Report (per class)")
            class_metrics["support"] = class_metrics["support"].astype(int)  # Asegura formato entero
            st.dataframe(
                class_metrics.style
                .background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])
                .format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"})
                .set_properties(**{"text-align": "center"})
            )


            # ─── CLASS-WISE METRICS CHART ────────────────────────────
            report_dict = classification_report(Y_test, Y_test_preds, target_names=categories, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose().iloc[:-3]  # Excluye promedio y soporte total

            import matplotlib.pyplot as plt
            fig_metric, ax_metric = plt.subplots(figsize=(10, 6))
            report_df[["precision", "recall", "f1-score"]].plot(kind='barh', ax=ax_metric)
            ax_metric.set_title("Classification Metrics by Class")
            ax_metric.set_xlabel("Score")
            ax_metric.set_xlim(0, 1.05)
            st.pyplot(fig_metric, use_container_width=True)


            # ─── EXPLORACIÓN DE CURVAS POR CLASE ───────────────────────
            from sklearn.metrics import roc_curve, precision_recall_curve, auc

            st.markdown("### 🎯 Explore ROC or Precision-Recall per Class")

            viz_type = st.radio(
                "Choose visualization type:", ["ROC Curve", "Precision-Recall Curve"],
                horizontal=True, key="viz_radio"
            )

            selected_label = st.selectbox(
                "Select a class to visualize:",
                options=[f"{i} - {label}" for i, label in enumerate(categories)],
                key="select_class"
            )

            # ✅ Extrae el índice de clase seleccionado
            class_idx = int(selected_label.split(" - ")[0])
            st.caption(f"Now viewing class **{class_idx} – {categories[class_idx]}**")

            # ── Preparación de datos
            y_true_binary = (Y_test == class_idx).astype(int)
            y_score_class = Y_test_probs[:, class_idx]

            fig, ax = plt.subplots(figsize=(6, 5))

            if viz_type == "ROC Curve":
                fpr, tpr, _ = roc_curve(y_true_binary, y_score_class)
                roc_auc = auc(fpr, tpr)

                ax.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"ROC Curve – {categories[class_idx]}")
                ax.legend(loc="lower right")

            else:
                precision, recall, _ = precision_recall_curve(y_true_binary, y_score_class)
                pr_auc = auc(recall, precision)

                ax.plot(recall, precision, color="darkorange", label=f"AUC = {pr_auc:.2f}")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(f"Precision-Recall Curve – {categories[class_idx]}")
                ax.legend(loc="lower left")

            st.pyplot(fig, use_container_width=True)


            # ─── IMPORTANCIA DE FEATURES ──────────────────────────────
            st.markdown("### Feature Importance")
            feature_names = vectorizer.get_feature_names_out()
            fig_feat = plt.figure(figsize=(10, 5))
            ax_feat  = fig_feat.add_subplot(111)
            skplt.estimators.plot_feature_importances(
                rf_classif,
                feature_names=feature_names,
                ax=ax_feat,
                title="Top Feature Importances"
            )
            st.pyplot(fig_feat, use_container_width=True)

            st.markdown("### Cumulative Feature Importance")
            importances = rf_classif.feature_importances_
            sorted_idx  = np.argsort(importances)[::-1]
            cum_imp     = np.cumsum(importances[sorted_idx])
            fig_cum = plt.figure(figsize=(8, 4))
            ax_cum  = fig_cum.add_subplot(111)
            ax_cum.plot(
                np.arange(1, len(cum_imp)+1),
                cum_imp,
                marker="o",
                label="Cumulative"
            )
            ax_cum.axhline(0.95, color="red", linestyle="--",
                           label="95% threshold")
            ax_cum.set_xlabel("Number of Features")
            ax_cum.set_ylabel("Cumulative Importance")
            ax_cum.set_title("Cumulative Feature Importance")
            ax_cum.legend()
            st.pyplot(fig_cum, use_container_width=True)
            
            # ─── PREDICCIONES ─────────────────────────────────────
            st.markdown("### 🔍 Prediction Results")

            # 1. Mostrar tabla con predicciones vs reales (sample)
            st.markdown("#### Sample Predictions (Real vs Predicted)")
            import pandas as pd

            pred_df = pd.DataFrame({
                "True Label": [categories[i] for i in Y_test[:15]],
                "Predicted Label": [categories[i] for i in Y_test_preds[:15]],
                "Correct": Y_test[:15] == Y_test_preds[:15]
            })

            st.dataframe(pred_df.style.applymap(
                lambda val: "color: green;" if val == True else "color: red;",
                subset=["Correct"]
            ))

            # 2. Mostrar distribución de clases predichas
            st.markdown("#### Predicted Class Distribution")
            from collections import Counter

            class_counts = Counter(Y_test_preds)
            dist_df = pd.DataFrame.from_dict(
                {categories[k]: v for k, v in class_counts.items()},
                orient="index",
                columns=["Count"]
            ).sort_values("Count", ascending=False)

            st.bar_chart(dist_df)
            
            # ─── COMPARACIÓN REAL VS PREDICHO ─────────────────────
            st.markdown("#### 📊 Real vs. Predicted Class Distribution")

            real_counts = pd.Series(Y_test).value_counts().sort_index()
            pred_counts = pd.Series(Y_test_preds).value_counts().sort_index()

            comparison_df = pd.DataFrame({
                "Actual": real_counts.values,
                "Predicted": pred_counts.values
            }, index=[categories[i] for i in real_counts.index])

            st.bar_chart(comparison_df)

            # 3. Mostrar predicción manual (opcional: campo input)
            st.markdown("#### Predict on a Custom Sample")
            sample_index = st.number_input("Choose a sample index from test set:", min_value=0, max_value=len(Y_test)-1, value=0)
            sample_text = datasets.fetch_20newsgroups(categories=categories).data[sample_index]

            st.markdown(f"**Sample Text:**\n\n{sample_text[:500]}...")  # mostrar solo primeros caracteres
            sample_vector = vectorizer.transform([sample_text])
            sample_pred = rf_classif.predict(sample_vector)[0]
            st.success(f"✅ Predicted Category: **{categories[sample_pred]}**")

elif selected == "Data Science Model":
    st.title("Decision Tree Classifier 🌳")
    st.markdown("Train a Decision Tree model using recruitment data.")

    X, y = load_recruitment_data()

    with st.form("dtree_form"):
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown("### 🧠 Training Parameters")
            max_depth = st.slider("Max Depth", 2, 20, value=8)
            min_samples_split = st.slider("Min Samples Split", 2, 10, value=4)
            criterion = st.selectbox("Criterion", options=["gini", "entropy", "log_loss"])
            train_button = st.form_submit_button("🚀 Train Decision Tree")

    if train_button:
        model, X_train, X_test, y_train, y_test = train_decision_tree(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion
        )

        st.session_state["dtree_model"] = model
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

        Y_train_preds = model.predict(X_train)
        Y_test_preds = model.predict(X_test)

        with col2:
            col21, col22 = st.columns(2, gap="medium")
            with col21:
                st.metric("Test Accuracy", value="{:.2f} %".format(100 * accuracy_score(y_test, Y_test_preds)))
            with col22:
                st.metric("Train Accuracy", value="{:.2f} %".format(100 * accuracy_score(y_train, Y_train_preds)))

            st.markdown("### Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4.5))  # Tamaño ajustado como Random Forest
            skplt.metrics.plot_confusion_matrix(y_test, Y_test_preds, ax=ax_cm)
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)

        # Mensaje fuera de col2 para que quede al final como en Random Forest
        st.success("✅ Model trained successfully with selected hyperparameters!")

    # Resultados post-entrenamiento
    if "dtree_model" in st.session_state:
        model = st.session_state["dtree_model"]
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        y_train = st.session_state["y_train"]
        y_test = st.session_state["y_test"]
        Y_test_preds = model.predict(X_test)

        st.markdown("## 📊 Model Results Overview")

        tab1, tab2, tab3, tab4, tab5, tab7, tab8, tab9, tab10 = st.tabs([
            "📋 Classification Report",
            "📈 ROC Curve",
            "🧠 Feature Importances",
            "❌ Error Analysis",
            "🔄 Cross-Validation",
            "📦 Probability Distribution",
            "⚖️ Bias vs Variance",
            "🧠 Assumptions & Limitations",
            "🏁 Threshold Tuning",
        ])

        with tab1:
            report_dict = classification_report(y_test, Y_test_preds, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose().round(2)

            summary_metrics = report_df.loc[["accuracy", "macro avg", "weighted avg"]]
            class_metrics = report_df.drop(["accuracy", "macro avg", "weighted avg"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Macro F1 Score", f"{summary_metrics.loc['macro avg', 'f1-score']:.2f}")
            with col2:
                st.metric("Overall Accuracy", f"{summary_metrics.loc['accuracy', 'precision']:.2f}")

            st.dataframe(
                class_metrics.style
                .background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])
                .format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"})
                .set_properties(**{"text-align": "center"})
            )

            main_class = class_metrics["f1-score"].idxmax()
            worst_class = class_metrics["f1-score"].idxmin()

            best_score = class_metrics.loc[main_class, "f1-score"]
            worst_score = class_metrics.loc[worst_class, "f1-score"]

            st.markdown("### 🧾 Interpretation")
            st.info(f"""
            - ✅ Best performance: **{main_class}** with F1 = {best_score:.2f}
            - ⚠️ Weakest performance: **{worst_class}** with F1 = {worst_score:.2f}
            - 📊 Macro Avg F1 Score: **{summary_metrics.loc['macro avg', 'f1-score']:.2f}**
            """)
            
            context = f"""
            Macro F1 Score: {summary_metrics.loc['macro avg', 'f1-score']:.2f}
            Overall Accuracy: {summary_metrics.loc['accuracy', 'precision']:.2f}
            Best Class: {main_class} with F1 = {best_score:.2f}
            Weakest Class: {worst_class} with F1 = {worst_score:.2f}
            """

            ai_insight_block(context, label_prefix="📋")

        with tab2:
            from sklearn.metrics import roc_curve, auc, roc_auc_score
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc_score = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_title("Receiver Operating Characteristic")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend()
            st.pyplot(fig_roc, use_container_width=True)

            st.markdown("### 💡 ROC Interpretation")
            if auc_score >= 0.9:
                interp = "✅ Excellent: the classifier distinguishes classes almost perfectly."
            elif auc_score >= 0.8:
                interp = "👍 Very Good: minor overlap between classes."
            elif auc_score >= 0.7:
                interp = "🟡 Acceptable, but improvement is needed."
            elif auc_score >= 0.6:
                interp = "⚠️ Low performance – model barely beats random."
            else:
                interp = "❌ Poor – model performance is not usable."

            st.info(f"AUC Score: **{auc_score:.2f}** — {interp}")

        with tab3:
            st.markdown("### 🧠 Feature Importances")
            importances = model.feature_importances_
            features = X_train.columns            
            importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)

            fig_feat, ax_feat = plt.subplots(figsize=(10, 5))
            ax_feat.barh(importance_df["Feature"], importance_df["Importance"])
            ax_feat.set_xlabel("Importance")
            ax_feat.set_title("Top Feature Importances")
            st.pyplot(fig_feat)
            top_feature = importance_df.iloc[0]["Feature"]
            top_value = importance_df.iloc[0]["Importance"]
            st.markdown("### 🧠 Interpretation")
            st.info(f"The most important feature is **{top_feature}** with an importance score of **{top_value:.2f}**. This means it contributes the most to the model’s decision-making. Consider validating this feature’s quality and consistency.")

        with tab4:
            st.markdown("### 🔍 Error Analysis – Correct vs Incorrect by Class")
            error_df = pd.DataFrame({"Actual": y_test, "Predicted": Y_test_preds})
            error_df["Correct"] = (error_df["Actual"] == error_df["Predicted"]).astype(int)

            import seaborn as sns
            fig_err, ax_err = plt.subplots()
            sns.countplot(data=error_df, x="Actual", hue="Correct", palette={1: "green", 0: "red"}, ax=ax_err)
            ax_err.set_title("Correct vs Incorrect by Class")
            st.pyplot(fig_err)
            error_rate = 1 - accuracy_score(y_test, Y_test_preds)
            st.markdown("### 🧾 Interpretation")
            st.info(f"Overall error rate: **{error_rate:.2%}**. Misclassifications are most frequent in the classes with lower support or class imbalance. Consider adding more examples or rebalancing.")

        with tab5:
            from sklearn.model_selection import cross_val_score
            st.markdown("### 🔄 Cross-Validation Scores (Accuracy)")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            st.write(f"**Mean Accuracy:** {cv_scores.mean():.2f}")
            st.write(f"**Std Dev:** {cv_scores.std():.2f}")

            fig_cv, ax_cv = plt.subplots()
            ax_cv.plot(range(1, 6), cv_scores, marker='o')
            ax_cv.set_xticks(range(1, 6))
            ax_cv.set_ylim(0, 1)
            ax_cv.set_xlabel("Fold")
            ax_cv.set_ylabel("Accuracy")
            ax_cv.set_title("CV Accuracy per Fold")
            st.pyplot(fig_cv)

            st.markdown("### 🔁 F1 Macro Score")
            f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
            st.metric("Mean F1 Score", f"{f1_scores.mean():.2f}")
            st.markdown("### 🧾 Interpretation")
            st.info(f"CV Accuracy: **{cv_scores.mean():.2f} ± {cv_scores.std():.2f}**. The model shows {'stable' if cv_scores.std() < 0.05 else 'unstable'} generalization performance across different subsets.")
            if f1_scores.std() > 0.05:
                st.warning("⚠️ High variability across folds.")
            else:
                st.success("✅ Consistent performance across folds.")

        with tab7:
            st.markdown("### 📦 Probability Distribution of Predictions")
            import seaborn as sns
            proba_preds = model.predict_proba(X_test)[:, 1]
            fig_prob, ax_prob = plt.subplots()
            sns.histplot(proba_preds, bins=20, kde=True, color="skyblue", ax=ax_prob)
            ax_prob.set_title("Distribution of Predicted Probabilities (Class = 1)")
            ax_prob.set_xlabel("Predicted Probability")
            st.pyplot(fig_prob, use_container_width=True)
            st.markdown("### 🧾 Interpretation")
            confident = (proba_preds > 0.9).mean()
            uncertain = ((proba_preds > 0.4) & (proba_preds < 0.6)).mean()
            st.info(f"**{confident:.1%}** of predictions are made with high confidence (>90%). About **{uncertain:.1%}** fall into the uncertainty zone (40–60%), suggesting possible threshold tuning or soft decision support.")

        with tab8:
            st.markdown("### ⚖️ Bias vs Variance Tradeoff")
            depths = list(range(1, 21))
            train_acc = []
            test_acc = []

            for d in depths:
                m = DecisionTreeClassifier(max_depth=d, random_state=42)
                m.fit(X_train, y_train)
                train_acc.append(m.score(X_train, y_train))
                test_acc.append(m.score(X_test, y_test))

            fig_bias, ax_bias = plt.subplots()
            ax_bias.plot(depths, train_acc, label="Train Accuracy", marker="o")
            ax_bias.plot(depths, test_acc, label="Test Accuracy", marker="s")
            ax_bias.set_title("Bias vs Variance Tradeoff Curve")
            ax_bias.set_xlabel("Max Depth")
            ax_bias.set_ylabel("Accuracy")
            ax_bias.legend()
            st.pyplot(fig_bias, use_container_width=True)
            st.markdown("### 🧾 Interpretation")
            optimal_depth = depths[np.argmax(test_acc)]
            st.info(f"The best generalization was achieved at **depth {optimal_depth}**, where test accuracy peaked. Lower depths underfit the model, while higher depths start overfitting.")

        with tab9:
            st.markdown("## 🧠 Assumptions & Limitations")
            st.markdown("""
            ### ✅ Assumptions
            - The features are representative of the decision-making process.
            - There is no multicollinearity (Decision Trees are robust but redundant features can cause instability).
            - Training data is correctly labeled and representative of future candidates.
            - Features have meaningful splits with sufficient granularity.

            ### ⚠️ Limitations
            - Susceptible to overfitting, especially with high depth.
            - Poor extrapolation with unseen categorical combinations.
            - Decision boundaries are axis-parallel, which may limit generalization.
            - Less effective on highly imbalanced datasets without pre-processing.

            🧾 **Recommendation**: Evaluate pruning techniques or consider ensemble methods (e.g., Random Forest) for better generalization.
            """)
            
        with tab10:
            st.markdown("## 🏁 Threshold Tuning")

            thresholds = np.arange(0.0, 1.01, 0.01)
            proba_preds = model.predict_proba(X_test)[:, 1]

            accuracy_scores = []
            precision_scores = []
            recall_scores = []

            for t in thresholds:
                preds = (proba_preds >= t).astype(int)
                accuracy_scores.append(accuracy_score(y_test, preds))
                precision_scores.append(np.round(np.sum((preds == 1) & (y_test == 1)) / max(np.sum(preds == 1), 1), 2))
                recall_scores.append(np.round(np.sum((preds == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1), 2))

            fig_thresh, ax_thresh = plt.subplots(figsize=(8, 5))
            ax_thresh.plot(thresholds, accuracy_scores, label="Accuracy")
            ax_thresh.plot(thresholds, precision_scores, label="Precision")
            ax_thresh.plot(thresholds, recall_scores, label="Recall")
            ax_thresh.set_title("Metrics at Different Thresholds")
            ax_thresh.set_xlabel("Threshold")
            ax_thresh.set_ylabel("Score")
            ax_thresh.legend()
            st.pyplot(fig_thresh, use_container_width=True)

            st.info("Use this chart to choose an optimal threshold depending on whether you prioritize precision, recall, or balanced performance.")

                
        # Predicción personalizada
        st.markdown("### 🎯 Predict Hiring Decision on Custom Input")

        with st.expander("🧾 Fill Candidate Features", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.slider("Age", 20, 50, 30, key="age_slider")
                gender = st.selectbox("Gender", options=["Male", "Female"], key="gender_select")
                education = st.selectbox("Education Level", options=[
                    "Bachelor's Type 1", "Bachelor's Type 2", "Master's", "PhD"
                ], key="edu_select")
                st.caption("""
                **Type 1**: Full university bachelor's degree.  
                **Type 2**: Online, technical, or short-cycle bachelor's program.
                """)
                experience = st.slider("Experience Years", 0, 15, 3, key="exp_slider")
                prev_companies = st.slider("Previous Companies", 1, 5, 2, key="prev_slider")
            with col_b:
                distance = st.slider("Distance from Company (km)", 1.0, 50.0, 10.0, key="dist_slider")
                interview_score = st.slider("Interview Score", 0, 100, 60, key="interview_slider")
                skill_score = st.slider("Skill Score", 0, 100, 70, key="skill_slider")
                personality_score = st.slider("Personality Score", 0, 100, 75, key="personality_slider")
                recruitment_strategy = st.selectbox("Recruitment Strategy", options=[
                    "Human Decision-Making", "AI & Automated Systems", "Outsourced / External Hiring"
                ], key="recruitment_select")

        if st.button("🔍 Predict Hiring Outcome", key="predict_outcome_button"):
            gender = {"Male": 0, "Female": 1}[gender]
            education = {
                "Bachelor's Type 1": 1,
                "Bachelor's Type 2": 2,
                "Master's": 3,
                "PhD": 4
            }[education]
            recruitment_strategy = {
                "Human Decision-Making": 1,
                "AI & Automated Systems": 2,
                "Outsourced / External Hiring": 3
            }[recruitment_strategy]

            sample_input = pd.DataFrame([[
                age, gender, education, experience, prev_companies,
                distance, interview_score, skill_score,
                personality_score, recruitment_strategy
            ]], columns=st.session_state["X_train"].columns)

            prediction = model.predict(sample_input)[0]
            label = "Hired" if prediction == 1 else "Not Hired"

            st.markdown("---")
            st.success(f"**Predicted Outcome:** {label}")
            
            # 🔄 Crear PDF Ejecutivo mejorado tras predicción manual
            from fpdf import FPDF
            import seaborn as sns
            import os

            # Calcular métricas visuales
            importances = model.feature_importances_
            feature_names = X_train.columns
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values("Importance", ascending=False)

            proba = model.predict_proba(sample_input)[0][1]

            # 1. Gráfico de importancias
            fig1, ax1 = plt.subplots(figsize=(5, 2.5))  # más pequeño
            sns.barplot(data=importance_df.head(5), x="Importance", y="Feature", ax=ax1, palette="Blues_d")
            ax1.set_title("Top 5 Most Important Features", fontsize=10)
            plt.tight_layout()
            plot1_path = "plot_feature_importance.png"
            fig1.savefig(plot1_path)

            # 2. Gráfico de probabilidad
            fig2, ax2 = plt.subplots(figsize=(5, 2.5))  # más pequeño
            sns.histplot(model.predict_proba(X_test)[:, 1], bins=20, kde=True, color="skyblue", ax=ax2)
            ax2.axvline(proba, color="red", linestyle="--", label=f"Candidate = {proba:.2f}")
            ax2.legend()
            ax2.set_title("Probability Distribution (Test Set)", fontsize=10)
            plt.tight_layout()
            plot2_path = "plot_probability_dist.png"
            fig2.savefig(plot2_path)
            
            # 3. Calibration Curve (Reliability)
            from sklearn.calibration import calibration_curve

            y_prob = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

            fig3, ax3 = plt.subplots(figsize=(5, 2.5))
            ax3.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
            ax3.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
            ax3.set_title("Calibration Curve (Reliability)", fontsize=10)
            ax3.set_xlabel("Predicted Probability")
            ax3.set_ylabel("True Probability")
            ax3.legend()
            plt.tight_layout()
            plot3_path = "plot_calibration_curve.png"
            fig3.savefig(plot3_path)

            # Crear PDF profesional
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(200, 10, "Candidate Hiring Decision Report", ln=1, align="C")

            # Sección 1 - Inputs
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, "Candidate Input Summary", ln=1)
            pdf.set_font("Arial", "", 11)
            for col, val in zip(sample_input.columns, sample_input.values[0]):
                pdf.cell(200, 8, f"{col}: {val}", ln=1)

            # Sección 2 - Model Decision
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, "Model Decision", ln=1)
            pdf.set_font("Arial", "", 11)
            pdf.cell(200, 8, f"Prediction: {'HIRED' if prediction == 1 else 'NOT HIRED'}", ln=1)
            pdf.cell(200, 8, f"Probability: {proba:.2%}", ln=1)
            pdf.ln(4)
            pdf.set_font("Arial", "I", 11)
            pdf.multi_cell(0, 8,
                "The following visualizations highlight the decision-making rationale of the model. "
                "They illustrate the top predictive features used to assess this candidate and the position of this candidate's probability "
                "within the overall distribution observed during testing. These insights support a more informed hiring decision."
            )

            # Gráficos
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, "Top Feature Importances", ln=1)
            pdf.image(plot1_path, x=15, w=180)
            pdf.ln(5)
            pdf.cell(200, 10, "Model Probability Distribution", ln=1)
            pdf.image(plot2_path, x=15, w=180)
            
            # Insertar al PDF
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, "Calibration Curve", ln=1)
            pdf.image(plot3_path, x=15, w=180)

            # NUEVA PÁGINA para resumen ejecutivo
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, "Executive Summary & Recommendation", ln=1)

            # Análisis AI (usando GPT-3.5)
            from openai import OpenAIError
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional HR Data Analyst creating executive-level summaries."},
                        {"role": "user", "content": f"""
            You are preparing an executive summary for a hiring decision based on the following candidate input:

            {sample_input.to_string(index=False)}

            Model output:
            - Prediction: {'Hired' if prediction == 1 else 'Not Hired'}
            - Probability: {proba:.2%}
            - Top Features: {importance_df.head(3).to_string(index=False)}

            Write a concise, polished, and professional executive report (1 page max, 3-4 paragraphs) for a CEO.
            """}
                    ]
                )
                summary_text = response.choices[0].message.content.strip()
            except OpenAIError as e:
                summary_text = "Executive summary not available due to API error."

            # Añadir al PDF
            pdf.set_font("Arial", "", 11)
            for line in summary_text.split("\n"):
                if line.strip():
                    pdf.multi_cell(0, 8, line)

            # Guardar PDF
            final_path = "Hiring_Report_Executive.pdf"
            pdf.output(final_path)

            with open(final_path, "rb") as f:
                st.download_button("📥 Download Executive PDF Report", f, file_name="Hiring_Report_Executive.pdf")

elif selected == "Models Comparison":
    st.title("📊 Model Comparison – Logistic vs Decision Tree")

    if "log_model" not in st.session_state or "dtree_model" not in st.session_state:
        st.warning("Please train both models first in their respective tabs.")
    else:
        log_model = st.session_state["log_model"]
        dt_model = st.session_state["dtree_model"]

        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        y_pred_log = log_model.predict(X_test)
        y_prob_log = log_model.predict_proba(X_test)[:, 1]

        y_pred_tree = dt_model.predict(X_test)
        y_prob_tree = dt_model.predict_proba(X_test)[:, 1]

        # --- Side-by-side metrics
       # Extrae dinámicamente los valores desde session_state
        logistic_train_acc = accuracy_score(st.session_state["y_train_log"], st.session_state["log_model"].predict(st.session_state["X_train_log"]))
        logistic_test_acc = accuracy_score(st.session_state["y_test_log"], st.session_state["log_model"].predict(st.session_state["X_test_log"]))
        tree_train_acc = accuracy_score(st.session_state["y_train"], st.session_state["dtree_model"].predict(st.session_state["X_train"]))
        tree_test_acc = accuracy_score(st.session_state["y_test"], st.session_state["dtree_model"].predict(st.session_state["X_test"]))

        # Crear tabla y gráfica
        import pandas as pd
        import matplotlib.pyplot as plt
        
        import numpy as np
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score, mean_squared_error, mean_absolute_error,
            roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
        )

        # Variables del usuario (ya cargadas del session_state en su app)
        log_model = st.session_state["log_model"]
        dt_model = st.session_state["dtree_model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        y_test_log = st.session_state["y_test_log"]
        y_test_tree = st.session_state["y_test"]
        y_probs_log = log_model.predict_proba(st.session_state["X_test_log"])[:, 1]
        y_probs_tree = dt_model.predict_proba(st.session_state["X_test"])[:, 1]
        residuals_log = y_test_log - y_probs_log
        residuals_tree = y_test_tree - y_probs_tree

        # Métricas cuantitativas
        mse_log = mean_squared_error(y_test_log, y_probs_log)
        mse_tree = mean_squared_error(y_test_tree, y_probs_tree)
        mae_log = mean_absolute_error(y_test_log, y_probs_log)
        mae_tree = mean_absolute_error(y_test_tree, y_probs_tree)

        fpr_log, tpr_log, _ = roc_curve(y_test_log, y_probs_log)
        fpr_tree, tpr_tree, _ = roc_curve(y_test_tree, y_probs_tree)
        auc_log = auc(fpr_log, tpr_log)
        auc_tree = auc(fpr_tree, tpr_tree)

        prec_log, rec_log, _ = precision_recall_curve(y_test_log, y_probs_log)
        prec_tree, rec_tree, _ = precision_recall_curve(y_test_tree, y_probs_tree)

        # Clasificación
        rep_log = classification_report(y_test_log, log_model.predict(st.session_state["X_test_log"]), output_dict=True)
        rep_tree = classification_report(y_test_tree, dt_model.predict(st.session_state["X_test"]), output_dict=True)

        # Interpretaciones automatizadas por bloque
        summary_texts = {
            "accuracy": f"""
        🔍 **Train-Test Accuracy Gap**:
        - Logistic Regression: {abs(accuracy_score(st.session_state["y_train_log"], log_model.predict(st.session_state["X_train_log"])) - accuracy_score(y_test_log, log_model.predict(st.session_state["X_test_log"]))):.2%}
        - Decision Tree: {abs(accuracy_score(st.session_state["y_train"], dt_model.predict(st.session_state["X_train"])) - accuracy_score(y_test_tree, dt_model.predict(st.session_state["X_test"]))):.2%}

        🔎 A smaller gap suggests better generalization. Logistic Regression appears more stable across datasets.
        """,
            "roc_curve": f"""
        🎯 **AUC Scores**:
        - Logistic: {auc_log:.2f}
        - Decision Tree: {auc_tree:.2f}

        { "✅ Logistic is more reliable in separating classes." if auc_log > auc_tree else "✅ Decision Tree shows slightly better separation capacity." }
        """,
            "classification": f"""
        📊 **Classification Metrics (Weighted Avg)**:
        - F1-score (Logistic): {rep_log['weighted avg']['f1-score']:.2f}
        - F1-score (Tree): {rep_tree['weighted avg']['f1-score']:.2f}

        { "📌 Logistic performs better overall." if rep_log['weighted avg']['f1-score'] > rep_tree['weighted avg']['f1-score'] else "📌 Decision Tree achieves slightly higher F1 performance." }
        """,
            "features": f"""
        🧠 **Feature Use Comparison**:
        - Logistic Coefficients reflect **linear influence**.
        - Tree Importances show **non-linear decision splits**.

        Combining both insights could yield a stronger ensemble.
        """,
            "precision_recall": f"""
        📈 **Precision-Recall Comparison**:
        Logistic: smoother transitions, consistent recall.

        Decision Tree: more fluctuation in precision.

        { "📌 Logistic may offer more stable decision-making under uncertainty." if rec_log.mean() > rec_tree.mean() else "📌 Decision Tree prioritizes high-recall regions." }
        """,
            "residuals": f"""
        📉 **Residual Error Distributions**:
        - Logistic Mean Residual: {np.mean(residuals_log):.2f}
        - Tree Mean Residual: {np.mean(residuals_tree):.2f}

        { "✅ Logistic residuals are more centered, suggesting better calibration." if abs(np.mean(residuals_log)) < abs(np.mean(residuals_tree)) else "⚠️ Decision Tree has more dispersed errors – risk of overfitting." }
        """
        }

        acc_data = {
            "Model": ["Logistic Regression", "Decision Tree"],
            "Train Accuracy (%)": [logistic_train_acc * 100, tree_train_acc * 100],
            "Test Accuracy (%)": [logistic_test_acc * 100, tree_test_acc * 100]
        }
        df_acc = pd.DataFrame(acc_data)

        # Mostrar tabla
        st.markdown("### 📋 Accuracy Comparison Table")
        st.dataframe(df_acc.style.format({"Train Accuracy (%)": "{:.2f}", "Test Accuracy (%)": "{:.2f}"}))

        # Gráfico
        fig_acc, ax_acc = plt.subplots()
        bar_width = 0.35
        index = range(len(df_acc))

        train_bars = ax_acc.bar(index, df_acc["Train Accuracy (%)"], bar_width, label="Train Accuracy", color="#90caf9")
        test_bars = ax_acc.bar([i + bar_width for i in index], df_acc["Test Accuracy (%)"], bar_width, label="Test Accuracy", color="#42a5f5")

        ax_acc.set_xlabel("Model")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title("Train vs Test Accuracy per Model")
        ax_acc.set_xticks([i + bar_width / 2 for i in index])
        ax_acc.set_xticklabels(df_acc["Model"])
        ax_acc.set_ylim(0, 110)
        ax_acc.legend()

        for bars in [train_bars, test_bars]:
            for bar in bars:
                height = bar.get_height()
                ax_acc.annotate(f'{height:.2f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')

        st.pyplot(fig_acc)
        st.markdown("### 📘 Interpretation")
        st.info(f"""
        🔍 **Train-Test Accuracy Gap**:
        - Logistic Regression: {abs(logistic_train_acc - logistic_test_acc):.2%}
        - Decision Tree: {abs(tree_train_acc - tree_test_acc):.2%}

        A smaller gap suggests better generalization. {'Logistic Regression appears more stable.' if abs(logistic_train_acc - logistic_test_acc) < abs(tree_train_acc - tree_test_acc) else 'Decision Tree may be overfitting.'}
        """)

        st.markdown("---")

        # --- Confusion Matrices
        st.subheader("📦 Confusion Matrix")

        cm1 = confusion_matrix(y_test, y_pred_log)
        cm2 = confusion_matrix(y_test, y_pred_tree)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Logistic Regression**")
            fig1, ax1 = plt.subplots()
            sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax1)
            st.pyplot(fig1)
        with col2:
            st.markdown("**Decision Tree**")
            fig2, ax2 = plt.subplots()
            sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=ax2)
            st.pyplot(fig2)
        
        st.info("A confusion matrix shows how predictions match actual values. More diagonal values mean better accuracy.")

        st.markdown("---")

        # --- ROC Curve Comparison
        from sklearn.metrics import roc_curve, auc

        fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
        fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
        auc_log = auc(fpr_log, tpr_log)
        auc_tree = auc(fpr_tree, tpr_tree)

        st.subheader("📈 ROC Curve Comparison")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr_log, tpr_log, label=f"Logistic (AUC = {auc_log:.2f})")
        ax_roc.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC = {auc_tree:.2f})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_title("ROC Curves")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc)
        st.markdown("### 📘 Interpretation")
        st.info(f"""
        🎯 **AUC Scores**:
        - Logistic: {auc_log:.2f}
        - Decision Tree: {auc_tree:.2f}

        {'✅ Logistic is more reliable in separating classes.' if auc_log > auc_tree else '✅ Decision Tree shows slightly better separation capacity.'}
        """)

        st.markdown("---")

        # --- Classification Report
        st.subheader("📋 Classification Metrics Comparison")
        rep_log = classification_report(y_test, y_pred_log, output_dict=True)
        rep_tree = classification_report(y_test, y_pred_tree, output_dict=True)

        comp_df = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1-score"],
            "Logistic": [
                rep_log["weighted avg"]["precision"],
                rep_log["weighted avg"]["recall"],
                rep_log["weighted avg"]["f1-score"]
            ],
            "Decision Tree": [
                rep_tree["weighted avg"]["precision"],
                rep_tree["weighted avg"]["recall"],
                rep_tree["weighted avg"]["f1-score"]
            ]
        }).round(2)
        
        st.markdown("### 📘 Interpretation")
        st.info(f"""
        📊 **Classification (Weighted Avg F1)**:
        - Logistic F1: {rep_log['weighted avg']['f1-score']:.2f}
        - Decision Tree F1: {rep_tree['weighted avg']['f1-score']:.2f}

        {'📌 Logistic performs better overall.' if rep_log['weighted avg']['f1-score'] > rep_tree['weighted avg']['f1-score'] else '📌 Decision Tree achieves slightly higher F1 performance.'}
        """)


        st.dataframe(comp_df.set_index("Metric").style.highlight_max(axis=1, color="lightgreen"))

        # --- Coefficients vs. Feature Importances
        st.subheader("🧠 Feature Insights")
        coef_df = pd.DataFrame({
            "Feature": st.session_state["X_train"].columns,
            "Logistic_Coefficient": log_model.coef_[0],
            "Tree_Importance": dt_model.feature_importances_
        })

        coef_df["Abs_Coeff"] = coef_df["Logistic_Coefficient"].abs()
        top_features = coef_df.sort_values("Abs_Coeff", ascending=False).head(10)

        fig_feat, ax_feat = plt.subplots(figsize=(10, 5))
        ax_feat.bar(top_features["Feature"], top_features["Abs_Coeff"], label="Logistic Coefficient", alpha=0.6)
        ax_feat.bar(top_features["Feature"], top_features["Tree_Importance"], label="Tree Importance", alpha=0.6)
        ax_feat.set_title("Top 10 Features – Coefficients vs Importances")
        ax_feat.set_xticklabels(top_features["Feature"], rotation=45)
        ax_feat.legend()
        st.pyplot(fig_feat)
        st.markdown("### 📘 Interpretation")
        st.info("""
        🧠 Coefficients show **linear relationships**, useful for interpretability.  
        🌳 Tree Importances reveal **non-linear decision paths**.

        Use both for model explainability or combine them in ensemble methods.
        """)

        
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error,
            roc_curve, auc, precision_recall_curve
        )

        # Recuperar datos
        y_test_log = st.session_state["y_test_log"]
        y_test_tree = st.session_state["y_test"]
        y_probs_log = st.session_state["log_model"].predict_proba(st.session_state["X_test_log"])[:, 1]
        y_probs_tree = st.session_state["dtree_model"].predict_proba(st.session_state["X_test"])[:, 1]

        # ───── Métricas cuantitativas ─────
        mse_log = mean_squared_error(y_test_log, y_probs_log)
        mse_tree = mean_squared_error(y_test_tree, y_probs_tree)
        mae_log = mean_absolute_error(y_test_log, y_probs_log)
        mae_tree = mean_absolute_error(y_test_tree, y_probs_tree)

        fpr_log, tpr_log, _ = roc_curve(y_test_log, y_probs_log)
        fpr_tree, tpr_tree, _ = roc_curve(y_test_tree, y_probs_tree)
        auc_log = auc(fpr_log, tpr_log)
        auc_tree = auc(fpr_tree, tpr_tree)

        prec_log, rec_log, _ = precision_recall_curve(y_test_log, y_probs_log)
        prec_tree, rec_tree, _ = precision_recall_curve(y_test_tree, y_probs_tree)

        # ───── Gráfica 2: Precision-Recall ─────
        st.subheader("📊 Precision-Recall Curve Comparison")
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(rec_log, prec_log, label="Logistic", linestyle="--")
        ax_pr.plot(rec_tree, prec_tree, label="Decision Tree")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision vs Recall")
        ax_pr.legend()
        st.pyplot(fig_pr)
        st.info("This curve highlights the balance between precision and recall. Steeper curves are preferred. Logistic has a smoother drop-off.")
        st.markdown("### 📘 Interpretation")
        st.info(f"""
        📈 **Precision vs Recall**:
        - Logistic shows smoother transitions.
        - Decision Tree has higher fluctuation.

        {'📌 Logistic offers more consistent decision-making under uncertainty.' if rec_log.mean() > rec_tree.mean() else '📌 Tree favors high-recall thresholds, useful for risk-averse decisions.'}
        """)

        # ───── Gráfica 3: Residual Distribution ─────
        st.subheader("📉 Residual Error Distribution")
        residuals_log = y_test_log - y_probs_log
        residuals_tree = y_test_tree - y_probs_tree

        fig_resid, ax_resid = plt.subplots()
        sns.kdeplot(residuals_log, label="Logistic", fill=True, ax=ax_resid)
        sns.kdeplot(residuals_tree, label="Decision Tree", fill=True, ax=ax_resid)
        ax_resid.set_title("Distribution of Residuals")
        ax_resid.set_xlabel("Residual (True - Predicted Probability)")
        ax_resid.legend()
        st.pyplot(fig_resid)
        st.info("Residuals close to 0 indicate well-calibrated models. Wider spread in Decision Tree may suggest overfitting.")
        st.markdown("### 📘 Interpretation")
        st.info(f"""
        📉 **Residuals (True - Probabilities)**:
        - Mean Logistic Residual: {np.mean(residuals_log):.2f}
        - Mean Tree Residual: {np.mean(residuals_tree):.2f}

        {'✅ Logistic residuals are tighter — better calibration.' if abs(np.mean(residuals_log)) < abs(np.mean(residuals_tree)) else '⚠️ Tree residuals are more spread — may indicate overfitting.'}
        """)

        # ───── Tabla resumen de métricas ─────
        import pandas as pd
        metrics_df = pd.DataFrame({
            "Model": ["Logistic Regression", "Decision Tree"],
            "Test AUC": [auc_log, auc_tree],
            "MSE": [mse_log, mse_tree],
            "MAE": [mae_log, mae_tree]
        }).round(3)

        st.subheader("🧾 Summary of Quantitative Metrics")
        st.dataframe(metrics_df)

        st.info("""
        - **MSE** (Mean Squared Error) penalizes large prediction errors.
        - **MAE** (Mean Absolute Error) gives an average of error magnitude.
        - **AUC** indicates classification strength — higher is better.
        Use this table to assess not just accuracy, but calibration, stability, and prediction confidence.
        """)

        # ───── Executive Summary Final ─────
        st.markdown("## 🧠 Executive Summary – Strategic Model Insights")

        summary_text = f"""
        ### 🔍 Comparative Overview

        - **Accuracy**: Logistic Regression achieved a test accuracy of **{logistic_test_acc*100:.2f}%**, while Decision Tree reached **{tree_test_acc*100:.2f}%**. However, the Decision Tree also showed a significantly higher train accuracy (**{tree_train_acc*100:.2f}%**) compared to Logistic (**{logistic_train_acc*100:.2f}%**), suggesting a higher risk of overfitting.

        - **Discriminative Power (AUC)**: Logistic scored **{auc_log:.2f}** and Decision Tree **{auc_tree:.2f}** in AUC — both are within strong performance ranges, with Decision Tree slightly outperforming.

        - **Calibration & Error Metrics**:
            - Logistic Regression has lower **MSE = {mse_log:.3f}** and **MAE = {mae_log:.3f}**, suggesting better probabilistic calibration.
            - Decision Tree produced **MSE = {mse_tree:.3f}** and **MAE = {mae_tree:.3f}**, indicating less precise probability estimates.

        - **Precision-Recall Tradeoff**: Logistic exhibits a smoother drop-off in precision-recall space, indicating more consistent predictions across thresholds.

        - **Residuals**: Logistic's residuals are more tightly distributed around 0, supporting its better calibration and generalization.

        ---

        ### 💡 Strategic Recommendations

        ✅ Use **Logistic Regression** when:
        - Interpretability is key (linear coefficients).
        - You need robustness and calibration.
        - You want to avoid overfitting in small datasets.

        🌳 Use **Decision Tree** when:
        - Capturing non-linear interactions is critical.
        - Explainability is required through tree visualization.
        - You're dealing with more complex decision boundaries.

        🤝 For robust results, consider:
        - **Ensembling both models** (e.g., Voting Classifier or Stacking).
        - **Hybrid approaches** where trees detect interactions and logits calibrate probabilities.

        ---
        """
        st.markdown(summary_text)
        
