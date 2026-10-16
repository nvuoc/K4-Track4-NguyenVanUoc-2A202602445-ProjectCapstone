// v2: Browser-owned video loop. Frames are sent directly to registered
// Python callbacks using Colab's supported invokeFunction API.
// No eval_js() lookup across output iframe boundaries.
(() => {
  const prior = document.getElementById('vin-dms-demo-v2');
  if (prior) prior.remove();
  const root = document.createElement('div');
  root.id = 'vin-dms-demo-v2';
  root.innerHTML = `
    <style>
      #vin-dms-demo-v2 {font:14px Arial,sans-serif;max-width:760px;background:#101d2c;color:#f2f6fc;border-radius:12px;padding:16px}
      #vin-dms-demo-v2 button {margin:4px;padding:9px 12px;border:0;border-radius:7px;cursor:pointer;background:#dce8fc;color:#18283b;font-weight:bold}
      #vin-dms-demo-v2 button:hover {background:#b6d0ff}
      #vin-dms-demo-v2 button:disabled {opacity:.55;cursor:not-allowed}
      #vin-dms-demo-v2 canvas {width:100%;max-width:640px;border-radius:8px;background:#23394f;display:block;margin:10px auto}
      #vin-dms-demo-v2 .muted {font-size:12px;color:#aec4d8;margin:8px 0}
      #vin-dms-demo-v2 .line {margin:7px 0;display:flex;align-items:center;flex-wrap:wrap}
      #vin-dms-demo-v2 .tag {margin-right:12px;font-size:13px}
    </style>
    <h3 style="margin:0 0 4px">VinFast Driver Monitoring · Colab webcam (v2)</h3>
    <p class="muted">Chạy Cell 1, Cell 2a, Cell 2b, Cell 3. Sau đó chỉ cần START webcam: AI tự xử lý. Nhìn thẳng trong 12 khung hình đầu để hiệu chuẩn.</p>
    <div class="line">
      <button id="vin-start">▶ START webcam + AI</button>
      <button id="vin-stop">■ STOP webcam</button>
      <button id="vin-calib">⟳ Hiệu chuẩn lại</button>
      <button id="vin-rec">● Quay video dự phòng</button>
    </div>
    <video id="vin-video" autoplay playsinline muted style="display:none"></video>
    <canvas id="vin-canvas" width="640" height="480"></canvas>
    <div class="line"><span class="tag">Nhãn thủ công để đánh giá:</span>
      <button class="vin-label" data-value="focus">1 · Tập trung</button>
      <button class="vin-label" data-value="talk">2 · Chỉ nói chuyện</button>
      <button class="vin-label" data-value="distracted">3 · Nói + quay đầu</button>
    </div>
    <p id="vin-message" class="muted">Chưa bật webcam. Không cần chạy một Cell 4 để xử lý realtime.</p>
    <p class="muted">Sau khi bấm STOP, chạy Cell 4 để xem số liệu và tải CSV. Không bấm Run all khi camera đang chạy.</p>`;
  document.body.appendChild(root);
  try { google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true); } catch (_) {}

  const video = root.querySelector('#vin-video');
  const canvas = root.querySelector('#vin-canvas');
  const ctx = canvas.getContext('2d');
  const grab = document.createElement('canvas');
  grab.width = 640; grab.height = 480;
  const grabCtx = grab.getContext('2d', {willReadFrequently:true});
  const msg = root.querySelector('#vin-message');
  const recordBtn = root.querySelector('#vin-rec');
  const startBtn = root.querySelector('#vin-start');
  const state = {running:false, gt:'unlabeled', prediction:{state:'WAITING'},
    reset:true, stream:null, recorder:null, chunks:[], startedAt:0,
    busy:false, sessionId:0};
  const labels = {
    WAITING:'CHỜ MÔ HÌNH', NO_FACE:'KHÔNG THẤY KHUÔN MẶT', CALIBRATING:'ĐANG HIỆU CHUẨN',
    FOCUSED:'ĐANG TẬP TRUNG', POSSIBLE_TALKING:'NGHI ĐANG NÓI CHUYỆN',
    LOOKING_AWAY:'ĐANG QUAY ĐẦU', TALK_AWAY_PENDING:'NÓI + QUAY ĐẦU',
    ALERT_AWAY:'CẢNH BÁO: QUAY ĐẦU LÂU', ALERT_TALK_AWAY:'CẢNH BÁO: NÓI + QUAY ĐẦU'
  };

  async function callPython(name, args=[]) {
    if (!window.google?.colab?.kernel?.invokeFunction) {
      throw new Error('Không tìm thấy cầu nối Colab. Hãy mở notebook trong Google Colab rồi chạy lại Cell 3.');
    }
    const response = await google.colab.kernel.invokeFunction(name, args, {});
    const data = response?.data?.['application/json'];
    if (!data) throw new Error('Python không trả JSON; xem lỗi callback trong Colab.');
    return data;
  }

  async function startCamera() {
    if (state.running || state.busy) return;
    state.busy = true;
    startBtn.disabled = true;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw Error('Không hỗ trợ getUserMedia.');
      msg.textContent = 'Đang khởi tạo AI trong Python...';
      await callPython('vinfast_dms.reset', []);
      state.stream = await navigator.mediaDevices.getUserMedia({
        video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false
      });
      video.srcObject = state.stream;
      await video.play();
      state.running = true;
      state.reset = true;
      state.prediction = {state:'CALIBRATING'};
      state.startedAt = performance.now();
      state.sessionId += 1;
      msg.textContent = 'Webcam + AI đã bật. Nhìn thẳng vào camera để hiệu chuẩn.';
      pump(state.sessionId);
    } catch (e) {
      if (state.stream) state.stream.getTracks().forEach(t=>t.stop());
      state.stream = null;
      state.running = false;
      msg.textContent = 'Lỗi START webcam/AI: ' + String(e);
    } finally {
      state.busy = false;
      startBtn.disabled = false;
    }
  }

  function stopCamera(reason) {
    state.running = false;
    state.sessionId += 1;
    if (state.recorder?.state === 'recording') state.recorder.stop();
    if (state.stream) state.stream.getTracks().forEach(t=>t.stop());
    state.stream = null;
    video.srcObject = null;
    msg.textContent = reason || 'Đã dừng. Chạy Cell 4 để xem số liệu và tải CSV.';
  }

  async function pump(mySession) {
    // The capture, Python inference and response run serially: no backpressure queue.
    while (state.running && mySession === state.sessionId) {
      if (performance.now() - state.startedAt > MAX_DEMO_SECONDS * 1000) {
        stopCamera('Đã hết thời gian demo. Chạy Cell 4 để xem log.');
        break;
      }
      if (video.readyState < 2) {
        await new Promise(r=>setTimeout(r,100));
        continue;
      }
      try {
        grabCtx.drawImage(video,0,0,640,480);
        const packet = {
          data_url:grab.toDataURL('image/jpeg',0.65),
          timestamp_ms:performance.now(), gt:state.gt, reset:state.reset
        };
        state.reset = false;
        const result = await callPython('vinfast_dms.frame', [packet]);
        if (state.running && mySession === state.sessionId) state.prediction = result;
      } catch (e) {
        stopCamera('Lỗi xử lý AI: '+String(e)+'; kiểm tra output của Cell 3.');
        break;
      }
      await new Promise(r=>setTimeout(r,220));
    }
  }

  function recordVideo() {
    if (state.recorder?.state === 'recording') {
      state.recorder.stop();recordBtn.textContent='● Quay video dự phòng';return;
    }
    if (!state.running) {msg.textContent='Hãy START webcam trước.';return;}
    try {
      if (!window.MediaRecorder || !canvas.captureStream) throw Error('Trình duyệt không hỗ trợ quay canvas.');
      const mime = ['video/webm;codecs=vp8','video/webm'].find(t=>MediaRecorder.isTypeSupported(t));
      const recorder = new MediaRecorder(canvas.captureStream(12),mime?{mimeType:mime}:{});
      state.chunks = [];
      recorder.ondataavailable = e=>{if(e.data?.size)state.chunks.push(e.data);};
      recorder.onstop=()=>{
        const url=URL.createObjectURL(new Blob(state.chunks,{type:'video/webm'}));
        const a=document.createElement('a'); a.href=url;a.download='vinfast_dms_demo_backup.webm';
        document.body.appendChild(a);a.click();a.remove();
        setTimeout(()=>URL.revokeObjectURL(url),10000);
        recordBtn.textContent='● Quay video dự phòng';
        msg.textContent='Đã tạo video .webm. Kiểm tra thư mục Downloads.';
      };
      state.recorder = recorder;
      recorder.start(1000);
      recordBtn.textContent='■ Dừng quay và lưu';
      msg.textContent='Đang quay video dự phòng có cảnh báo trên canvas.';
    } catch(e) {msg.textContent='Lỗi quay video: '+String(e);}
  }

  root.querySelector('#vin-start').onclick=startCamera;
  root.querySelector('#vin-stop').onclick=()=>stopCamera();
  root.querySelector('#vin-calib').onclick=()=>{
    state.reset=true;state.prediction={state:'CALIBRATING'};
    msg.textContent='Đang yêu cầu hiệu chuẩn lại. Nhìn thẳng vào camera.';
  };
  root.querySelector('#vin-rec').onclick=recordVideo;
  root.querySelectorAll('.vin-label').forEach(btn=>btn.onclick=()=>{
    state.gt=btn.dataset.value;
    root.querySelectorAll('.vin-label').forEach(b=>b.style.background=b===btn?'#a8f0b7':'#dce8fc');
    msg.textContent='Nhãn thủ công: '+state.gt+' (nhãn không điều khiển dự đoán).';
  });

  function draw() {
    if (state.running && video.readyState >= 2) {
      ctx.save();ctx.translate(640,0);ctx.scale(-1,1);ctx.drawImage(video,0,0,640,480);ctx.restore();
    } else {
      ctx.fillStyle='#26394d';ctx.fillRect(0,0,640,480);
      ctx.fillStyle='#b4c7d9';ctx.font='24px Arial';ctx.fillText('Webcam preview',205,250);
    }
    const p=state.prediction||{};
    const danger=!!p.alert;
    const amber=['POSSIBLE_TALKING','LOOKING_AWAY','TALK_AWAY_PENDING'].includes(p.state);
    const accent=danger?'#ff616a':amber?'#ffcc60':'#66e2af';
    if (p.bbox) {
      const [x0,y0,x1,y1]=p.bbox;
      ctx.strokeStyle=accent;ctx.lineWidth=3;
      ctx.strokeRect((1-x1)*640,y0*480,(x1-x0)*640,(y1-y0)*480);
      if (p.points) for (const [x,y] of p.points) {
        ctx.beginPath();ctx.arc((1-x)*640,y*480,3,0,Math.PI*2);
        ctx.fillStyle=accent;ctx.fill();
      }
    }
    ctx.fillStyle='rgba(9,23,37,.85)';ctx.fillRect(0,0,640,84);
    ctx.fillStyle=accent;ctx.font='bold 22px Arial';
    ctx.fillText(labels[p.state]||p.state||'CHỜ MÔ HÌNH',14,30);
    ctx.fillStyle='#f7fafc';ctx.font='14px Arial';
    ctx.fillText(`FPS ${Number(p.fps||0).toFixed(1)}  |  Inference ${Number(p.inference_ms||0).toFixed(0)} ms  |  Cảnh báo ${p.episodes||0}`,14,54);
    ctx.fillText(`Môi Δ ${Number(p.mouth_variation||0).toFixed(3)}  |  Hướng Δ ${Number(p.yaw_delta||0).toFixed(2)}  |  Nhãn ${state.gt}`,14,75);
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();
