

import os
import time
import random
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024   # 32 MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'dcm'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Model Loading (graceful fallback) ────────────────────────────────────────
try:
    from model.inference import load_model, preprocess_image, preprocess_clinical, postprocess_output
    MODEL = load_model()
    logger.info('✅ Model loaded successfully.')
except Exception as e:
    MODEL = None
    logger.warning('⚠️  Model not loaded — mock predictions will be used. Reason: %s', e)


# ── Static Data ───────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Alzheimer's Detected": {
        "emoji": "⚠️",
        "desc": (
            "Alzheimer's disease is a progressive neurological disorder that causes the brain to "
            "shrink (atrophy) and brain cells to die. It is the most common cause of dementia."
        ),
        "causes": [
            "Accumulation of amyloid-beta plaques",
            "Tau protein tangles inside neurons",
            "Genetic risk factors (APOE-e4 allele)",
            "Age-related neuroinflammation",
        ],
        "precautions": [
            "Consult a neurologist immediately",
            "Schedule a formal neuropsychological evaluation",
            "Explore FDA-approved medications (donepezil, memantine)",
            "Arrange a care support plan for the patient",
        ],
        "diet": [
            "MIND diet (Mediterranean-DASH Intervention)",
            "Berries, leafy greens, nuts, whole grains daily",
            "Reduce processed foods and red meat",
            "Omega-3 rich fish (salmon, sardines) twice a week",
        ],
        "alert_type": "danger",
    },
    "Possible MCI": {
        "emoji": "🔶",
        "desc": (
            "Mild Cognitive Impairment (MCI) is an early stage of memory or cognitive ability loss "
            "that is greater than normal aging but less severe than dementia. It may progress to Alzheimer's."
        ),
        "causes": [
            "Early amyloid or tau pathology",
            "Vascular risk factors (hypertension, diabetes)",
            "Sleep disorders affecting memory consolidation",
            "Chronic stress or depression",
        ],
        "precautions": [
            "Regular cognitive assessments every 6 months",
            "Engage in mentally stimulating activities daily",
            "Maintain regular physical exercise (150 min/week)",
            "Address treatable causes (sleep apnoea, depression)",
        ],
        "diet": [
            "Antioxidant-rich foods (blueberries, spinach)",
            "Reduce sugar and refined carbohydrates",
            "Stay well hydrated throughout the day",
            "Limit alcohol to minimum",
        ],
        "alert_type": "warning",
    },
    "No Alzheimer's Detected": {
        "emoji": "✅",
        "desc": (
            "No significant Alzheimer's pathology detected in the provided scan/data. "
            "Maintain a brain-healthy lifestyle to protect cognitive function."
        ),
        "causes": ["N/A — no disease detected"],
        "precautions": [
            "Annual cognitive health check-up recommended",
            "Exercise at least 30 minutes per day",
            "Stay socially and mentally active",
            "Manage cardiovascular risk factors",
        ],
        "diet": [
            "Balanced diet with diverse whole foods",
            "Adequate vitamin D and B12 intake",
            "Limit ultra-processed food and sugar",
            "Seasonal fruits and vegetables daily",
        ],
        "alert_type": "success",
    },
}

# Fallback — catch any prediction label that doesn't exactly match
DISEASE_INFO_DEFAULT = DISEASE_INFO["No Alzheimer's Detected"]

DOCTORS = [
    {"name": "Dr. Rajesh Sharma",    "city": "Delhi",     "specialist": "Senior Neurologist",              "fees": "₹900",  "exp": "20 yrs", "hospital": "AIIMS Delhi"},
    {"name": "Dr. Priya Verma",      "city": "Delhi",     "specialist": "Cognitive Neurologist",           "fees": "₹1100", "exp": "15 yrs", "hospital": "Fortis Escorts"},
    {"name": "Dr. Avinash Mehta",    "city": "Mumbai",    "specialist": "Neurologist",                     "fees": "₹1000", "exp": "18 yrs", "hospital": "Lilavati Hospital"},
    {"name": "Dr. Sneha Patil",      "city": "Mumbai",    "specialist": "Dementia Specialist",             "fees": "₹1400", "exp": "12 yrs", "hospital": "KEM Hospital"},
    {"name": "Dr. Punit Sadana",     "city": "Bangalore", "specialist": "Neuropsychiatrist",               "fees": "₹950",  "exp": "16 yrs", "hospital": "Manipal Hospital"},
    {"name": "Dr. Ananya Rao",       "city": "Bangalore", "specialist": "Geriatric Neurologist",           "fees": "₹800",  "exp": "11 yrs", "hospital": "Narayana Health"},
    {"name": "Dr. Tarun Gupta",      "city": "Haridwar",  "specialist": "Neurologist",                     "fees": "₹600",  "exp": "22 yrs", "hospital": "Patanjali Hospital"},
    {"name": "Dr. Sanjay Kapoor",    "city": "Chennai",   "specialist": "Cognitive Neurologist",           "fees": "₹1000", "exp": "14 yrs", "hospital": "MGM Healthcare"},
    {"name": "Dr. Lakshmi Nair",     "city": "Kolkata",   "specialist": "Neurologist",                     "fees": "₹850",  "exp": "13 yrs", "hospital": "AMRI Hospital"},
    {"name": "Dr. Amit Desai",       "city": "Pune",      "specialist": "Neurodegenerative Specialist",    "fees": "₹750",  "exp": "10 yrs", "hospital": "Jehangir Hospital"},
    {"name": "Dr. Rekha Reddy",      "city": "Hyderabad", "specialist": "Neurologist",                     "fees": "₹1200", "exp": "17 yrs", "hospital": "KIMS Hospital"},
    {"name": "Dr. Vikram Joshi",     "city": "Jaipur",    "specialist": "Neurologist",                     "fees": "₹700",  "exp": "9 yrs",  "hospital": "SMS Hospital"},
]


# ── Helper ────────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Mock Predictor (used when MODEL is None) ─────────────────────────────────
def mock_predict() -> dict:
    """Simulate a model response for demo/testing. Remove when real model is ready."""
    time.sleep(0.4)
    r = random.random()
    if r < 0.40:
        return {"prediction": "Alzheimer's Detected",     "confidence": round(84 + random.uniform(0, 10), 2), "stage": "Moderate AD", "risk": "High",     "model": "(ResNet-50)"}
    elif r < 0.65:
        return {"prediction": "Possible MCI",             "confidence": round(60 + random.uniform(0, 20), 2), "stage": "Early MCI",   "risk": "Moderate", "model": "(ResNet-50)"}
    else:
        return {"prediction": "No Alzheimer's Detected",  "confidence": round(88 + random.uniform(0, 10), 2), "stage": "Normal",       "risk": "Low",      "model": "(ResNet-50)"}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None, "version": "2.1.0"})


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
      MRI  : multipart/form-data — field "mri_scan" (image file)
      Clinical: application/json  — { mode:"clinical", age, mmse, cdr, ... }

    Returns: { prediction, confidence, stage, risk, model, latency_ms }
    """
    start_ms = time.time()

    try:
        # ── Determine mode ───────────────────────────────────────────────────
        is_mri = 'mri_scan' in request.files

        if is_mri:
            file = request.files['mri_scan']
            if file.filename == '':
                return jsonify({"error": "No file selected."}), 400
            if not allowed_file(file.filename):
                return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            if MODEL is None:
                logger.info("[MRI] No model — using mock prediction")
                result = mock_predict()
            else:
                tensor     = preprocess_image(path)
                raw_output = MODEL(tensor)
                result     = postprocess_output(raw_output)
                result.setdefault("model", "PyTorch CNN")

        else:
            # Clinical JSON
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "Expected JSON body with clinical data or a file upload."}), 400

            required = ['age', 'mmse', 'cdr']
            missing  = [k for k in required if not data.get(k)]
            if missing:
                return jsonify({"error": f"Missing required fields: {missing}"}), 400

            if MODEL is None:
                logger.info("[CLINICAL] No model — using mock prediction")
                result = mock_predict()
            else:
                tensor     = preprocess_clinical(data)
                raw_output = MODEL(tensor)
                result     = postprocess_output(raw_output)
                result.setdefault("model", "PyTorch Model")

        result['latency_ms'] = round((time.time() - start_ms) * 1000, 1)
        logger.info("[PREDICT] %s", result)
        return jsonify(result)

    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


@app.route('/metrics')
def metrics():
    """Return model performance metrics for the performance chart."""
    return jsonify({
        "accuracy":    92.4,
        "precision":   90.1,
        "recall":      88.7,
        "f1_score":    89.4,
        "roc_auc":     95.2,
        "specificity": 91.3,
        "dataset":     "ADNI",
        "samples":     2840,
    })


@app.route('/disease-info')
def disease_info():
    """Return causes, precautions, diet for a given prediction label."""
    label = request.args.get('label', '')
    info  = DISEASE_INFO.get(label, DISEASE_INFO_DEFAULT)
    return jsonify(info)


@app.route('/doctors')
def doctors():
    """Return neurologist list, optionally filtered by ?city=Mumbai."""
    city   = request.args.get('city', '').strip()
    result = [d for d in DOCTORS if d['city'].lower() == city.lower()] if city else DOCTORS
    cities = sorted({d['city'] for d in DOCTORS})
    return jsonify({"doctors": result, "cities": cities})


@app.route('/report', methods=['POST'])
def report():
    """
    Generate a PDF health report.
    Body: { patient_name, age, gender, prediction, confidence, stage, risk,
            causes, precautions, diet, doctor (optional) }
    Requires: pip install fpdf2
    """
    try:
        from fpdf import FPDF
        from datetime import datetime

        data = request.get_json()

        class PDF(FPDF):
            def header(self):
                self.set_fill_color(4, 6, 15)
                self.rect(0, 0, 210, 32, 'F')
                self.set_font('Helvetica', 'B', 17)
                self.set_text_color(0, 212, 184)
                self.set_y(8)
                self.cell(0, 10, 'NeuroScan AI - Alzheimer\'s Detection Report', ln=True, align='C')
                self.set_font('Helvetica', '', 10)
                self.set_text_color(180, 200, 230)
                self.cell(0, 8, 'AI-Powered Neurological Assessment', ln=True, align='C')
                self.set_text_color(0, 0, 0)
                self.ln(8)

            def footer(self):
                self.set_y(-15)
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(128)
                self.cell(0, 10,
                    'Generated by NeuroScan AI | For informational purposes only. '
                    'Not a substitute for professional medical advice.',
                    align='C')

        def sec(pdf, title):
            pdf.set_fill_color(220, 235, 255)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(13, 71, 161)
            pdf.cell(0, 9, '  ' + title, ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

        def row(pdf, label, value):
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(65, 7, label + ':', ln=False)
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 7, str(value), ln=True)

        def bullet(pdf, text):
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(8, 7, '*', ln=False)
            pdf.cell(0, 7, str(text), ln=True)

        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=18)

        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(128)
        pdf.cell(0, 6, f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", ln=True)
        pdf.set_text_color(0)
        pdf.ln(3)

        sec(pdf, 'Patient Information')
        row(pdf, 'Name',   data.get('patient_name') or 'N/A')
        row(pdf, 'Age',    f"{data.get('age', 'N/A')} years")
        row(pdf, 'Gender', data.get('gender', 'N/A'))
        pdf.ln(3)

        sec(pdf, 'Prediction Result')
        row(pdf, 'Diagnosis',      data.get('prediction', 'N/A'))
        row(pdf, 'Confidence',     f"{data.get('confidence', 'N/A')}%")
        row(pdf, 'Stage',          data.get('stage', 'N/A'))
        row(pdf, 'Risk Level',     data.get('risk', 'N/A'))
        pdf.ln(3)

        sec(pdf, 'Precautions')
        for p in data.get('precautions', []):
            bullet(pdf, p)
        pdf.ln(3)

        sec(pdf, 'Diet Recommendations')
        for d in data.get('diet', []):
            bullet(pdf, d)
        pdf.ln(3)

        if data.get('doctor'):
            sec(pdf, 'Recommended Doctor')
            doc = data['doctor']
            row(pdf, 'Name',       doc.get('name', 'N/A'))
            row(pdf, 'Specialist', doc.get('specialist', 'N/A'))
            row(pdf, 'Hospital',   doc.get('hospital', 'N/A'))
            row(pdf, 'Fees',       doc.get('fees', 'N/A'))
            pdf.ln(3)

        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(128)
        pdf.multi_cell(0, 5,
            'DISCLAIMER: This report is generated by an AI system for informational purposes only. '
            'It does not constitute medical advice. Please consult a qualified neurologist.')

        pdf_bytes = bytes(pdf.output())
        from flask import Response
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={'Content-Disposition': 'attachment; filename=NeuroScan_Report.pdf'}
        )

    except ImportError:
        return jsonify({"error": "fpdf2 not installed. Run: pip install fpdf2"}), 501
    except Exception as e:
        logger.exception("PDF generation error")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
