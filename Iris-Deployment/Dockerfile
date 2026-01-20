FROM python:3.9

WORKDIR /app

# We install specific versions to avoid crashes
RUN pip install --no-cache-dir numpy scikit-learn joblib flask werkzeug

# کپی کردن کدها
COPY . .

# --- این بخش جادویی است ---
# اجرای کد آموزش دقیقاً قبل از اجرای سرور، تا فایل مدل داخل خود داکر ساخته شود
# THE MAGIC STEP: Train the model INSIDE the container
RUN python train.py

EXPOSE 80

CMD ["python", "serve_model.py"]