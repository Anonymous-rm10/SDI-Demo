const $ = id => document.getElementById(id);
const mic = $('mic'), status = $('status'), messages = $('messages');
let ws, context, stream, capture, source, muted;
let recording = false, starting = false, currentTurn = null, assistant = null;
let nextPlayAt = 0, sources = new Set(), playbackTurn = null, workletLoaded = false;
let reconnectTimer;

function send(event) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(event));
}
function log(value) {
  $('events').textContent = (JSON.stringify(value) + '\n' + $('events').textContent).slice(0, 6000);
}
function message(role, text) {
  const node = document.createElement('div');
  node.className = 'message ' + role;
  const label = document.createElement('span');
  label.className = 'label';
  label.textContent = ({user:'You', assistant:'Assistant', error:'Notice'})[role];
  const content = document.createElement('span');
  content.textContent = text;
  node.append(label, content);
  messages.append(node);
  while (messages.children.length > 200) messages.firstChild.remove();
  messages.scrollTop = messages.scrollHeight;
  return content;
}
async function audioContext() {
  context ||= new AudioContext();
  await context.resume();
  return context;
}
function stopPlayback() {
  const old = playbackTurn;
  playbackTurn = null;
  for (const item of sources) { item.onended = null; item.stop(); item.disconnect(); }
  sources.clear();
  nextPlayAt = 0;
  if (old !== null) send({type:'playback.end', turn_id:old});
}
function play(event) {
  if (!context || context.state !== 'running') {
    status.textContent = 'Enable the microphone or send a message to activate audio.';
    return;
  }
  const bytes = Uint8Array.from(atob(event.audio), c => c.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const samples = new Float32Array(bytes.length / 2);
  for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(i * 2, true) / 32768;
  if (!samples.length) return;
  const buffer = context.createBuffer(1, samples.length, event.sample_rate);
  buffer.copyToChannel(samples, 0);
  const item = context.createBufferSource();
  item.buffer = buffer;
  item.connect(context.destination);
  const turn = event.turn_id;
  if (playbackTurn !== turn) {
    playbackTurn = turn;
    send({type:'playback.start', turn_id:turn});
  }
  nextPlayAt = Math.max(context.currentTime + 0.025, nextPlayAt);
  item.start(nextPlayAt);
  nextPlayAt += buffer.duration;
  sources.add(item);
  status.textContent = 'Assistant is speaking';
  item.onended = () => {
    sources.delete(item);
    item.disconnect();
    if (!sources.size && playbackTurn === turn) {
      playbackTurn = null;
      send({type:'playback.end', turn_id:turn});
      status.textContent = recording ? 'Listening' : 'Waiting for input';
    }
  };
}
async function stopMic(commit = true) {
  recording = false;
  capture?.disconnect(); source?.disconnect(); muted?.disconnect();
  if (capture) capture.port.onmessage = null;
  stream?.getTracks().forEach(track => track.stop());
  capture = source = muted = stream = null;
  mic.textContent = 'Enable microphone'; mic.classList.remove('recording');
  if (commit) send({type:'audio.commit'});
  status.textContent = 'Microphone is off';
}
mic.onclick = async () => {
  if (starting) return;
  if (recording) return stopMic();
  starting = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      throw new Error('Microphone access requires localhost or HTTPS and a browser with AudioWorklet support.');
    }
    await audioContext();
    stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, echoCancellation:true,
      noiseSuppression:true, autoGainControl:true}});
    if (ws?.readyState !== WebSocket.OPEN) throw new Error('Connection closed');
    if (!workletLoaded) {
      await context.audioWorklet.addModule('/static/capture-worklet.js');
      workletLoaded = true;
    }
    source = context.createMediaStreamSource(stream);
    capture = new AudioWorkletNode(context, 'pcm-capture');
    muted = context.createGain(); muted.gain.value = 0;
    capture.port.onmessage = event => {
      if (!recording || ws?.readyState !== WebSocket.OPEN) return;
      if (ws.bufferedAmount > 65536) {
        stopMic(false);
        message('error', 'Connection is too slow. The microphone was paused; enable it again to resume.');
        return;
      }
      ws.send(event.data);
    };
    source.connect(capture); capture.connect(muted); muted.connect(context.destination);
    recording = true;
    mic.textContent = 'Disable microphone'; mic.classList.add('recording');
    status.textContent = 'Listening';
  } catch (error) {
    await stopMic(false);
    message('error', error.message);
  } finally { starting = false; }
};
$('interrupt').onclick = () => { stopPlayback(); send({type:'interrupt'}); status.textContent = 'Stopped'; };
$('reset').onclick = () => {
  stopPlayback(); send({type:'reset'}); messages.replaceChildren(); assistant = null; currentTurn = null;
};
$('form').onsubmit = async event => {
  event.preventDefault();
  const text = $('text').value.trim();
  if (!text || ws?.readyState !== WebSocket.OPEN) return;
  await audioContext();
  send({type:'text', text}); $('text').value = '';
};
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => {
    $('connection').textContent = 'Connected'; status.textContent = 'Waiting for input';
    for (const id of ['mic','interrupt','reset','send']) $(id).disabled = false;
  };
  ws.onmessage = ({data}) => {
    const event = JSON.parse(data);
    if (!['audio.chunk','text.delta','vad.start','vad.end'].includes(event.type)) log(event);
    switch (event.type) {
      case 'ready':
        $('connection').textContent = event.mock ? 'Connected · mock mode (test tone)' : 'Connected · local models'; break;
      case 'transcript': message('user', event.text); status.textContent = 'Checking turn intent'; break;
      case 'turn.start':
        stopPlayback(); currentTurn = event.turn_id; assistant = message('assistant', '');
        status.textContent = 'Generating a reply'; break;
      case 'text.delta':
        if (event.turn_id === currentTurn && assistant) {
          assistant.textContent += event.text; messages.scrollTop = messages.scrollHeight;
        } break;
      case 'audio.chunk': if (event.turn_id === currentTurn) play(event); break;
      case 'turn.cancelled':
        if (event.turn_id === currentTurn) { stopPlayback(); currentTurn = null; }
        break;
      case 'turn.failed':
        if (event.turn_id === currentTurn) { stopPlayback(); currentTurn = null; }
        break;
      case 'turn.done':
        if (!sources.size) status.textContent = recording ? 'Listening' : 'Waiting for input'; break;
      case 'input.ignored': status.textContent = event.reason === 'keep_speaking' ? 'Continuing the current reply' : 'Waiting for the next utterance'; break;
      case 'error': message('error', event.message); status.textContent = 'See the message above'; break;
      case 'vad.start': status.textContent = 'Speech detected'; break;
      case 'vad.end': status.textContent = 'Transcribing'; break;
    }
  };
  ws.onclose = () => {
    stopMic(false); stopPlayback(); currentTurn = null;
    $('connection').textContent = 'Disconnected · reconnecting with a new session';
    for (const id of ['mic','interrupt','reset','send']) $(id).disabled = true;
    clearTimeout(reconnectTimer); reconnectTimer = setTimeout(connect, 2000);
  };
  ws.onerror = () => { $('connection').textContent = 'Connection error'; };
}
connect();
window.addEventListener('pagehide', () => {
  clearTimeout(reconnectTimer); ws.onclose = null; stopMic(false); stopPlayback(); ws.close(); context?.close();
});
