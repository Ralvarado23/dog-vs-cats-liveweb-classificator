let model = null;
let videoStream = null;
let isStreaming = false;
let rafId = null;
let lastPredictAt = 0;
const PREDICT_INTERVAL_MS = 300;
let isTabVisible = true;

const elements = {
  modelStatus: document.getElementById('modelStatus'),
  statusText: document.getElementById('statusText'),
  predictionSection: document.getElementById('predictionSection'),
  resultsContainer: document.getElementById('resultsContainer'),
  dogResult: document.getElementById('dogResult'),
  catResult: document.getElementById('catResult'),
  dogPercentage: document.getElementById('dogPercentage'),
  catPercentage: document.getElementById('catPercentage'),
  confidenceNote: document.getElementById('confidenceNote'),
  workCanvas: document.getElementById('workCanvas'),
  video: document.getElementById('video'),
  videoWrapper: document.getElementById('videoWrapper'),
  cameraPrompt: document.getElementById('cameraPrompt')
};

function setModelStatus(type, message) {
  const dot = elements.modelStatus.querySelector('.status-dot');
  elements.modelStatus.className = `model-status ${type}`;
  dot.className = `status-dot ${type}`;
  elements.statusText.textContent = message;
}

async function loadModel() {
  try {
    setModelStatus('loading', 'Cargando modelo...');
    try {
      model = await tf.loadLayersModel('modelo/model.json');
    } catch {
      model = await tf.loadGraphModel('modelo/model.json');
    }
    setModelStatus('ready', 'Modelo listo');
    // Warmup: predicción dummy para evitar lag inicial
    try {
      const { width, height } = getModelInputSize();
      const warmup = tf.zeros([1, height, width, 3]);
      const out = model.predict(warmup);
      (Array.isArray(out) ? out : [out]).forEach(t => t.dispose());
      warmup.dispose();
    } catch {}
  } catch (error) {
    console.error('Error cargando modelo:', error);
    setModelStatus('error', 'Error al cargar modelo');
  }
}

function getModelInputSize() {
  try {
    const shape = model?.inputs?.[0]?.shape;
    const height = shape?.[1] || 120;
    const width = shape?.[2] || 120;
    return { width, height };
  } catch {
    return { width: 120, height: 120 };
  }
}

async function preprocessImageFromVideo() {
  const { width, height } = getModelInputSize();
  elements.workCanvas.width = width;
  elements.workCanvas.height = height;
  const ctx = elements.workCanvas.getContext('2d');
  const vw = elements.video.videoWidth || elements.video.clientWidth;
  const vh = elements.video.videoHeight || elements.video.clientHeight;
  const side = Math.min(vw, vh);
  const sx = Math.floor((vw - side) / 2);
  const sy = Math.floor((vh - side) / 2);
  ctx.drawImage(elements.video, sx, sy, side, side, 0, 0, width, height);
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(elements.workCanvas);
    return tensor.toFloat().expandDims(0);
  });
}

function interpretPrediction(output) {
  const data = output.dataSync();
  if (data.length === 1) {
    const dogProb = data[0];
    const catProb = 1 - dogProb;
    return { dogProb, catProb };
  } else if (data.length === 2) {
    return { catProb: data[0], dogProb: data[1] };
  } else {
    return { catProb: data[0] || 0, dogProb: data[1] || 0 };
  }
}

async function predictOnce() {
  if (!model || !isStreaming || !isTabVisible) return;
  try {
    const inputTensor = await preprocessImageFromVideo();
    let prediction = model.predict(inputTensor);
    const output = Array.isArray(prediction) ? prediction[0] : prediction;
    const { dogProb, catProb } = interpretPrediction(output);
    const dogPercent = Math.round(dogProb * 100);
    const catPercent = Math.round(catProb * 100);

    elements.dogPercentage.textContent = `${dogPercent}%`;
    elements.catPercentage.textContent = `${catPercent}%`;
    elements.resultsContainer.classList.add('show');
    elements.dogResult.classList.toggle('winner', dogProb > catProb);
    elements.catResult.classList.toggle('winner', catProb >= dogProb);

    const maxProb = Math.max(dogProb, catProb);
    const confidenceText = maxProb > 0.9 ? '🎯 Muy Alta' :
                          maxProb > 0.7 ? '✅ Alta' :
                          maxProb > 0.6 ? '🤔 Media' :
                          '❓ Baja';
    elements.confidenceNote.textContent = confidenceText;
    elements.confidenceNote.classList.add('show');

    inputTensor.dispose();
    ;(Array.isArray(prediction) ? prediction : [prediction]).forEach(p => p.dispose());
  } catch (error) {
    console.error('Error en predicción:', error);
  }
}

function predictionLoop(ts) {
  if (!isStreaming || !isTabVisible) return;
  if (ts === undefined) {
    rafId = requestAnimationFrame(predictionLoop);
    return;
  }
  if (ts - lastPredictAt >= PREDICT_INTERVAL_MS) {
    lastPredictAt = ts;
    predictOnce().finally(() => {
      rafId = requestAnimationFrame(predictionLoop);
    });
  } else {
    rafId = requestAnimationFrame(predictionLoop);
  }
}

async function startCamera() {
  try {
    if (isStreaming) return;
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280, max: 1920 },
        height: { ideal: 720, max: 1080 },
        frameRate: { ideal: 15, max: 30 }
      }
    });
    elements.video.srcObject = videoStream;
    await elements.video.play();
    elements.cameraPrompt.style.display = 'none';
    elements.video.classList.add('show');
    elements.videoWrapper.classList.add('reveal');
    isStreaming = true;
    elements.predictionSection.classList.add('show');
    document.querySelector('.content').classList.add('live');
    elements.resultsContainer.classList.add('show');

    // Ajuste manual del tamaño eliminado; CSS (aspect-ratio + object-fit) controla el layout

    // Iniciar rAF
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(predictionLoop);
  } catch (err) {
    console.error('Error con la cámara:', err);
    alert('Error accediendo a la cámara: ' + err.message);
  }
}

// Visibility API: pausar/reanudar predicciones
document.addEventListener('visibilitychange', () => {
  isTabVisible = document.visibilityState === 'visible';
  if (!isTabVisible) {
    cancelAnimationFrame(rafId);
  } else if (isStreaming) {
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(predictionLoop);
  }
});

// Arrancar cámara al hacer clic en el prompt
elements.cameraPrompt.addEventListener('click', startCamera);

// Inicializar
loadModel();
