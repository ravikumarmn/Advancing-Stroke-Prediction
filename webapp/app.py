import os
from flask import Flask, render_template, request, redirect
from webapp.helper import generate_ct_image, predict_skin_disease, classify_ct_or_mri, predict_stroke_disease

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    skin_classification_done = False
    predicted_label = None
    ct_image = None
    ct_mri_predicted_label = None
    stroke_predicted_class = None

    if request.method == 'POST':
        if 'skin_image' in request.files:
            skin_image = request.files['skin_image']
            if skin_image.filename == '':
                return redirect(request.url)
            
            skin_disease_model_path = "checkpoints/skin_disease/skin_disease_cnn.pth"
            skin_image_path = os.path.join("webapp/uploads", skin_image.filename)
            skin_image.save(skin_image_path)
            
            predicted_label = predict_skin_disease(skin_disease_model_path, skin_image_path)
            skin_classification_done = True
            
        elif 'mri_image' in request.files:
            mri_image = request.files['mri_image']
            if mri_image.filename == '':
                return redirect(request.url)
            
            mri_image_path = os.path.join("webapp/uploads", mri_image.filename)
            mri_image.save(mri_image_path)
            
            ct_image_full_path = generate_ct_image(mri_image_path)
            ct_image = "ct_image.png"
            
            ct_mri_checkpoint = "checkpoints/ct_mri_classifier/ct_mri_classifier_best_model.pth"
            predicted_class = classify_ct_or_mri(ct_mri_checkpoint, ct_image_full_path)
            ct_mri_predicted_label = predicted_class
            
            if ct_mri_predicted_label == "CT":
                stroke_model_path = "checkpoints/stroke_classifier/stroke_classifier_cnn.pth"
                stroke_predicted_class = predict_stroke_disease(model_path=stroke_model_path, image_path=ct_image_full_path)
                    
    nav_links = [
        {'url': '#skin-section', 'label': 'Skin Disease Classifier'},
        {'url': '#ct-mri-section', 'label': 'CT/MRI Image Generator'}
    ]
    
    return render_template('index.html', ct_image=ct_image, skin_classification_done=skin_classification_done, predicted_label=predicted_label, ct_mri_predicted_label=ct_mri_predicted_label, stroke_class=stroke_predicted_class, nav_links=nav_links)

if __name__ == '__main__':
    app.run(debug=True)
