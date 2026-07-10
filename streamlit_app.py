import os
import uuid
import json
import logging
from PIL import Image
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    THUMBNAILS_DIR, DB_PATH, CATEGORIES, DEFAULT_CONFIDENCE_THRESHOLD,
    MODEL_PATH, CLASS_INDICES_PATH, ACCURACY_CURVE_PATH, LOSS_CURVE_PATH,
    CONFUSION_MATRIX_PATH, CLASS_THRESHOLDS_PATH, TRAIN_DIR, VAL_DIR,
    DEFAULT_MARGIN_THRESHOLD
)
import database
import predict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("streamlit_app")

# Ensure database is initialized
database.init_db()

# Page configuration (matching custom styling)
st.set_page_config(
    page_title="SmartCatalog AI - Catalog Moderation Console",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Shopify/Meesho admin look
st.markdown("""
<style>
    /* Clean Marketplace Catalog Admin Theme overrides */
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
        height: 0px;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Ensure high contrast text on light background across systems */
    .stApp {
        background-color: #f9fafb !important;
        color: #1f2937 !important;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        color: #1f2937 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top bar branding */
    .top-bar {
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #e5e7eb;
    }
    .app-name {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827 !important;
        margin: 0;
        padding: 0;
        line-height: 1.2;
    }
    .app-tagline {
        font-size: 0.875rem;
        color: #6b7280 !important;
        margin: 0.25rem 0 0 0;
        padding: 0;
    }
    
    /* Headers & Text colors */
    .section-title {
        font-size: 1.125rem;
        font-weight: 500;
        color: #374151 !important;
        margin-bottom: 1rem;
        text-align: left;
    }
    
    /* Card layout elements */
    .admin-card-inner {
        padding: 0.5rem 0;
    }
    .prediction-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .prediction-category {
        font-size: 1.75rem;
        font-weight: 600;
        color: #0f766e !important;
        margin-bottom: 0.75rem;
        line-height: 1.1;
        text-align: left;
    }
    
    /* Status Badge Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.625rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1;
    }
    .pill-approved {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
    }
    .pill-review {
        background-color: #fef3c7 !important;
        color: #92400e !important;
    }
    .pill-reviewed {
        background-color: #dbeafe !important;
        color: #1e40af !important;
    }
    
    /* Slim Progress Bar */
    .slim-bar-container {
        background-color: #e5e7eb;
        border-radius: 9999px;
        height: 6px;
        width: 100%;
        margin: 0.5rem 0 1rem 0;
        overflow: hidden;
    }
    .slim-bar-fill {
        background-color: #0f766e;
        height: 100%;
        border-radius: 9999px;
    }
    .confidence-text {
        font-size: 0.8125rem;
        color: #374151 !important;
        font-weight: 500;
    }
    
    /* Inline Notification Styles (replaces default Streamlit alerts) */
    .custom-alert {
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 450;
        margin-bottom: 1rem;
        border: 1px solid transparent;
        text-align: left;
    }
    .alert-success {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
        border-color: #a7f3d0;
    }
    .alert-info {
        background-color: #e0f2fe !important;
        color: #0369a1 !important;
        border-color: #bae6fd;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to cache model loading
@st.cache_resource
def load_model_cached():
    model, classes = predict.load_inference_model()
    return model, classes

# Get training and validation split counts per class
def get_sample_counts() -> pd.DataFrame:
    # 1. Try to read cached split stats (useful for cloud deployments where raw dataset is gitignored)
    stats_path = os.path.join(os.path.dirname(MODEL_PATH), "dataset_stats.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r") as f:
                stats_data = json.load(f)
            rows = []
            for cat in CATEGORIES:
                cat_stats = stats_data.get(cat, {"train": 0, "val": 0})
                t_count = cat_stats.get("train", 0)
                v_count = cat_stats.get("val", 0)
                rows.append({
                    "Category": cat,
                    "Training Images": t_count,
                    "Validation Images": v_count,
                    "Ratio (Train/Val)": f"{t_count}:{v_count}" if v_count > 0 else "N/A"
                })
            return pd.DataFrame(rows)
        except Exception as e:
            logger.warning(f"Failed to load dataset_stats.json: {e}")

    # 2. Dynamic directory scanning fallback (for local development)
    rows = []
    for cat in CATEGORIES:
        t_dir = os.path.join(TRAIN_DIR, cat)
        v_dir = os.path.join(VAL_DIR, cat)
        t_count = len(os.listdir(t_dir)) if os.path.exists(t_dir) else 0
        v_count = len(os.listdir(v_dir)) if os.path.exists(v_dir) else 0
        rows.append({
            "Category": cat,
            "Training Images": t_count,
            "Validation Images": v_count,
            "Ratio (Train/Val)": f"{t_count}:{v_count}" if v_count > 0 else "N/A"
        })
    return pd.DataFrame(rows)

# Load per-class thresholds if available
def load_class_thresholds() -> dict:
    if os.path.exists(CLASS_THRESHOLDS_PATH):
        try:
            with open(CLASS_THRESHOLDS_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load per-class thresholds: {e}")
    return {}

try:
    model, classes = load_model_cached()
    model_loaded = os.path.exists(MODEL_PATH)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model_loaded = False
    classes = CATEGORIES

# Top Bar Header (App Name + Tagline)
st.markdown("""
<div class="top-bar">
    <h1 class="app-name">SmartCatalog AI</h1>
    <p class="app-tagline">Product Catalog Moderation & Routing Console</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Settings & Navigation Menu (Clean, no emojis)
with st.sidebar:
    st.markdown("<h3 style='color:#0f766e; margin-bottom:1rem; font-size:1.1rem; font-weight:600;'>Workspace View</h3>", unsafe_allow_html=True)
    app_mode = st.radio(
        "Navigation Options",
        ["Dashboard", "Review Queue", "History & Analytics", "Model Diagnostics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<h3 style='color:#0f766e; margin-bottom:1rem; font-size:1.1rem; font-weight:600;'>Settings</h3>", unsafe_allow_html=True)
    
    class_thresholds = load_class_thresholds()
    if class_thresholds:
        threshold_mode = st.radio(
            "Threshold Mode",
            ["Per-Class (Optimized)", "Manual Global Slider"]
        )
    else:
        threshold_mode = "Manual Global Slider"
        
    if threshold_mode == "Manual Global Slider":
        threshold = st.slider(
            "Auto-Approval Threshold",
            min_value=0.50,
            max_value=0.99,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.01,
            help="Confidence level below which product listings are sent to human moderators."
        )
    else:
        threshold = DEFAULT_CONFIDENCE_THRESHOLD
        st.info("Using precision-optimized per-class thresholds:")
        for cat, val in class_thresholds.items():
            st.write(f"**{cat}**: {val * 100:.1f}%")
            
    st.markdown("---")
    st.markdown("<h3 style='color:#0f766e; margin-bottom:1rem; font-size:1.1rem; font-weight:600;'>Ambiguity Detection</h3>", unsafe_allow_html=True)
    margin_threshold = st.slider(
        "Min Class Margin",
        min_value=0.00,
        max_value=0.50,
        value=DEFAULT_MARGIN_THRESHOLD,
        step=0.01,
        help="Minimum required difference between Top-1 and Top-2 probabilities. If smaller, the prediction is routed to human review."
    )
    
    st.markdown("---")
    st.markdown("<h3 style='color:#991b1b; margin-bottom:1rem; font-size:1.1rem; font-weight:600;'>System Maintenance</h3>", unsafe_allow_html=True)
    confirm_reset = st.checkbox("Confirm prediction database reset", value=False, help="Check this box to unlock the Reset button.")
    if st.button("Reset Marketplace Data", use_container_width=True, disabled=not confirm_reset, help="Clears all prediction logs and deletes all local thumbnails."):
        try:
            # Clear database logs
            database.reset_db()
            # Clear physical thumbnail images
            if os.path.exists(THUMBNAILS_DIR):
                for file in os.listdir(THUMBNAILS_DIR):
                    file_path = os.path.join(THUMBNAILS_DIR, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            st.toast("Prediction logs and thumbnails reset successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")


# Page 1: Dashboard (Predictor)
if app_mode == "Dashboard":
    st.markdown("<h2 class='section-title'>Automated Category Prediction</h2>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])
    
    # Left column: Image Upload using st.container(border=True)
    with col_left:
        with st.container(border=True):
            st.markdown("<h3 style='font-size:1rem; font-weight:600; margin-bottom:0.75rem;'>Upload Product Image</h3>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drag and drop image file here...",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                label_visibility="collapsed"
            )
            
            # Displays the image if uploaded
            if uploaded_file is not None:
                try:
                    img = Image.open(uploaded_file)
                    st.image(img, caption="Product Preview", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to open image file: {e}")
                    img = None
            else:
                img = None
                
    # Right column: Automated Prediction Result
    with col_right:
        if uploaded_file is not None and img is not None:
            with st.spinner("Classifying catalog category..."):
                try:
                    preds = predict.predict_image(img, model=model, top_k=3)
                    top_class, top_conf = preds[0]
                    second_class, second_conf = preds[1] if len(preds) > 1 else (None, 0.0)
                    margin = top_conf - second_conf
                    
                    # Routing checks
                    if threshold_mode == "Per-Class (Optimized)":
                        class_thresh = class_thresholds.get(top_class, DEFAULT_CONFIDENCE_THRESHOLD)
                    else:
                        class_thresh = threshold
                        
                    is_approved = (top_conf >= class_thresh) and (margin >= margin_threshold)
                    status = "Auto-Approved" if is_approved else "Manual Review"
                    
                    # Save local product thumbnail
                    os.makedirs(THUMBNAILS_DIR, exist_ok=True)
                    thumb_uuid = str(uuid.uuid4())
                    thumb_filename = f"thumb_{thumb_uuid}.png"
                    thumb_save_path = os.path.join(THUMBNAILS_DIR, thumb_filename)
                    
                    # Convert to RGB, resize and save
                    thumb_img = img.copy().convert("RGB")
                    thumb_img.thumbnail((128, 128))
                    thumb_img.save(thumb_save_path, "PNG")
                    db_thumb_path = f"outputs/thumbnails/{thumb_filename}"
                    
                    # Database log entry
                    pred_id = database.log_prediction(
                        image_name=uploaded_file.name,
                        image_path=db_thumb_path,
                        predicted_category=top_class,
                        confidence=float(top_conf),
                        status=status
                    )
                    
                    # Layout styles
                    badge_class = "pill-approved" if is_approved else "pill-review"
                    if is_approved:
                        badge_text = "Auto-Approved"
                    else:
                        if top_conf < class_thresh:
                            badge_text = "Requires Review"
                        else:
                            badge_text = f"Ambiguous ({margin*100:.0f}% Margin)"
                    
                    # Single styled result card containing thumbnail, predicted class, and confidence progress
                    with st.container(border=True):
                        st.markdown("<div class='admin-card-inner'><div class='prediction-title'>Prediction Summary</div></div>", unsafe_allow_html=True)
                        
                        col_card_thumb, col_card_info = st.columns([1, 2])
                        with col_card_thumb:
                            st.image(db_thumb_path, use_container_width=True, caption="Thumbnail")
                        with col_card_info:
                            st.markdown(f"""
                            <div class='prediction-category'>{top_class}</div>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <span class='confidence-text'>Confidence: {top_conf * 100:.1f}%</span>
                                <span class='status-pill {badge_class}'>{badge_text}</span>
                            </div>
                            <div class='slim-bar-container'>
                                <div class='slim-bar-fill' style='width: {top_conf * 100}%;'></div>
                            </div>
                            <div style='font-size: 0.75rem; color: #6b7280;'>Database Record ID: #{pred_id}</div>
                            """, unsafe_allow_html=True)
                        
                    # Expander for top-3 predictions detail below the main card
                    with st.expander("Show Detailed Probabilities"):
                        for cat, conf in preds:
                            col_txt, col_bar = st.columns([1, 2])
                            with col_txt:
                                st.write(f"**{cat}** ({conf * 100:.1f}%)")
                            with col_bar:
                                st.progress(float(conf))
                                
                except Exception as e:
                    st.error(f"Model prediction pipeline error: {e}")
                    logger.error(f"Inference error: {e}")
        else:
            st.markdown("<div class='custom-alert alert-info'>Upload a product photo on the left panel to execute catalog classification.</div>", unsafe_allow_html=True)

# Page 2: Review Queue
elif app_mode == "Review Queue":
    st.markdown("<h2 class='section-title'>Human Review Queue</h2>", unsafe_allow_html=True)
    st.write(
        "Catalog queue containing low-confidence predictions. Submit manual corrections "
        "to resolve product categories. Corrections are logged in a separate field for audit "
        "and retraining purposes."
    )
    
    # Fetch pending items
    pending_records, total_pending = database.get_history(status_filter="Manual Review", limit=100)
    
    if total_pending == 0:
        st.markdown("<div class='custom-alert alert-success'>Manual Review queue is empty. All items are successfully approved or resolved.</div>", unsafe_allow_html=True)
    else:
        # Form selection
        record_labels = [
            f"ID #{r['id']} - Image: {r['image_name']} - Predicted: {r['predicted_category']} ({r['confidence']*100:.1f}%)"
            for r in pending_records
        ]
        selected_label = st.selectbox("Select low-confidence prediction to review:", record_labels)
        
        if selected_label:
            idx = record_labels.index(selected_label)
            selected_record = pending_records[idx]
            
            st.markdown("---")
            col_rev_img, col_rev_form = st.columns([1, 1.2])
            
            with col_rev_img:
                # Load physical thumbnail path
                thumb_rel_path = selected_record["image_path"]
                base_dir = os.path.dirname(os.path.abspath(__file__))
                full_thumb_path = os.path.join(base_dir, thumb_rel_path)
                
                if os.path.exists(full_thumb_path):
                    st.image(Image.open(full_thumb_path), caption="Catalog Product Image", use_container_width=True)
                else:
                    st.warning(f"Product thumbnail path not found on disk: {thumb_rel_path}")
                    
            with col_rev_form:
                with st.container(border=True):
                    st.markdown("<h3 style='font-size:1.1rem; font-weight:600; margin-bottom:0.75rem;'>Review Catalog Entry</h3>", unsafe_allow_html=True)
                    st.write(f"**Item ID:** `{selected_record['id']}`")
                    st.write(f"**Filename:** `{selected_record['image_name']}`")
                    st.write(f"**Predicted Category:** `{selected_record['predicted_category']}`")
                    st.write(f"**Confidence:** `{selected_record['confidence']*100:.1f}%`")
                    st.write(f"**Timestamp:** {selected_record['timestamp']}")
                    
                    with st.form("resolve_form", clear_on_submit=True):
                        # Preselect the model predicted category
                        default_idx = CATEGORIES.index(selected_record['predicted_category']) if selected_record['predicted_category'] in CATEGORIES else 0
                        corrected_category = st.selectbox(
                            "Correct Category Classification:",
                            CATEGORIES,
                            index=default_idx
                        )
                        
                        submit_resolution = st.form_submit_button("Approve / Correct Category")
                        
                        if submit_resolution:
                            success = database.update_review(selected_record["id"], corrected_category)
                            if success:
                                st.toast("Resolution saved successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to write update to the database.")

# Page 3: History & Analytics
elif app_mode == "History & Analytics":
    st.markdown("<h2 class='section-title'>Moderation History & Audit Trail</h2>", unsafe_allow_html=True)
    
    # Render stats charts
    stats = database.get_stats()
    
    if stats["total"] == 0:
        st.markdown("<div class='custom-alert alert-info'>No moderation history found in the database. Upload files on the dashboard to log prediction records.</div>", unsafe_allow_html=True)
    else:
        col_ch1, col_ch2 = st.columns([1, 1])
        
        with col_ch1:
            st.markdown("<h4 style='font-size:1rem; font-weight:600; margin-bottom:0.5rem;'>Category Distribution</h4>", unsafe_allow_html=True)
            dist_data = stats["category_distribution"]
            if dist_data:
                df_dist = pd.DataFrame(list(dist_data.items()), columns=["Category", "Count"])
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=df_dist, x="Category", y="Count", color="#0f766e", ax=ax)
                ax.set_ylabel("Product Count")
                ax.set_xlabel("Category")
                plt.xticks(rotation=15)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No distribution stats to show.")
                
        with col_ch2:
            st.markdown("<h4 style='font-size:1rem; font-weight:600; margin-bottom:0.5rem;'>Routing Composition</h4>", unsafe_allow_html=True)
            # Fetch status counts
            conn = database.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT status, COUNT(*) FROM predictions GROUP BY status")
            status_data = dict(cursor.fetchall())
            conn.close()
            
            if status_data:
                df_status = pd.DataFrame(list(status_data.items()), columns=["Status", "Count"])
                fig, ax = plt.subplots(figsize=(6, 4))
                # Soft matching colors matching badges
                colors = ['#10b981' if k == 'Auto-Approved' else '#f59e0b' if k == 'Manual Review' else '#3b82f6' for k in df_status["Status"]]
                ax.pie(df_status["Count"], labels=df_status["Status"], autopct='%1.1f%%', colors=colors, startangle=140, 
                       wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No composition stats to show.")
                
        st.markdown("---")
        st.markdown("<h3 style='font-size:1.1rem; font-weight:600; margin-bottom:0.75rem;'>Database Log Audit Trail</h3>", unsafe_allow_html=True)
        
        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            cat_filter = st.selectbox("Filter by Category:", ["All"] + CATEGORIES)
        with col_f2:
            status_filter = st.selectbox("Filter by Status:", ["All", "Auto-Approved", "Manual Review", "Reviewed"])
            
        cat_param = None if cat_filter == "All" else cat_filter
        status_param = None if status_filter == "All" else status_filter
        
        history_records, total_records = database.get_history(category_filter=cat_param, status_filter=status_param, limit=1000)
        
        if total_records == 0:
            st.warning("No records matched the filter criteria.")
        else:
            df_history = pd.DataFrame(history_records)
            df_display = df_history.copy()
            
            # Map status values to include indicator dots in data grid
            status_map = {
                "Auto-Approved": "🟢 Auto-Approved",
                "Manual Review": "🟡 Requires Review",
                "Reviewed": "🔵 Reviewed"
            }
            df_display["status"] = df_display["status"].map(lambda x: status_map.get(x, x))
            
            # Multiply confidence by 100 for proper NumberColumn formatting
            df_display["confidence"] = df_display["confidence"] * 100
            
            # Format dataframe columns
            df_display = df_display[[
                "id", "timestamp", "image_name", "predicted_category", 
                "corrected_category", "confidence", "status"
            ]]
            
            st.dataframe(
                df_display,
                column_config={
                    "id": st.column_config.NumberColumn("Record ID", format="%d"),
                    "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                    "image_name": "Image Name",
                    "predicted_category": "AI Prediction",
                    "corrected_category": "Correction",
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.2f%%"),
                    "status": "Status"
                },
                use_container_width=True,
                hide_index=True
            )
            
            # CSV Download
            csv_bytes = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export History to CSV",
                data=csv_bytes,
                file_name="smartcatalog_moderation_logs.csv",
                mime="text/csv"
            )

# Page 4: Model Diagnostics
elif app_mode == "Model Diagnostics":
    st.markdown("<h2 class='section-title'>Model Health Diagnostics</h2>", unsafe_allow_html=True)
    st.write(
        "Performance indicators computed from training validation runs. Review training curves "
        "and confusion matrices to inspect visual overlapping catalog segments."
    )
    
    # Surface dataset counts transparently above metrics tabs
    with st.container(border=True):
        st.markdown("<h3 style='font-size:1rem; font-weight:600; margin-bottom:0.5rem;'>Dataset Split Distribution Summary</h3>", unsafe_allow_html=True)
        df_counts = get_sample_counts()
        st.dataframe(df_counts, use_container_width=True, hide_index=True)
        
    st.markdown("---")
    
    if not model_loaded:
        st.warning("No trained model weights (.pt) found on disk. Run train_model.py first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Epoch Metrics", "Confusion Matrix", "Classification Report"])
        
        with tab1:
            col_cur1, col_cur2 = st.columns(2)
            with col_cur1:
                if os.path.exists(ACCURACY_CURVE_PATH):
                    st.image(ACCURACY_CURVE_PATH, caption="Accuracy Curve", use_container_width=True)
                else:
                    st.info("Accuracy curve plot not found.")
            with col_cur2:
                if os.path.exists(LOSS_CURVE_PATH):
                    st.image(LOSS_CURVE_PATH, caption="Loss Curve", use_container_width=True)
                else:
                    st.info("Loss curve plot not found.")
                    
        with tab2:
            if os.path.exists(CONFUSION_MATRIX_PATH):
                st.image(CONFUSION_MATRIX_PATH, caption="Raw & Normalized Matrices", use_container_width=True)
            else:
                st.info("Confusion matrix plot not found.")
                
            # Dynamic confusion explanations
            analysis_json_path = os.path.join(os.path.dirname(CONFUSION_MATRIX_PATH), "confusion_analysis.json")
            if os.path.exists(analysis_json_path):
                try:
                    with open(analysis_json_path, "r") as f:
                        analysis_data = json.load(f)
                    st.info(analysis_data["analysis_text"])
                except Exception as e:
                    logger.warning(f"Could not load confusion explanation: {e}")
                    
        with tab3:
            csv_report_path = os.path.join(os.path.dirname(CONFUSION_MATRIX_PATH), "classification_report.csv")
            if os.path.exists(csv_report_path):
                df_rep = pd.read_csv(csv_report_path, index_col=0)
                st.dataframe(df_rep, use_container_width=True)
            else:
                st.info("Classification report CSV not found.")
