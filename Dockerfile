FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /lab
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/setup_pp.py
# Windows PowerShell (tắt CVAT Docker khi chạy model nếu WSL2 RAM <= 4 GB):
#   docker run --rm -v "C:\path\Data-VF:/data" -v "${PWD}\work:/work" -w /work <image> python /lab/scripts/pp_selftest.py --data /data
#   docker run --rm -v "C:\path\Data-VF:/data" -v "${PWD}\work:/work" -w /work <image> python /lab/scripts/vf_cvat.py day14 --student 001 --seed <SEED> --data /data
# Kết nối CVAT trên máy host từ trong container dùng: --host http://host.docker.internal:8080
CMD ["python", "scripts/pp_selftest.py"]
