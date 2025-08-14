/*
  Generative flowers on canvas (headless mode ready)
  - Draws rotating petaled flowers as dreamy background.
  - Exposes window.flowersControl to control via gestures:
    setRotationSpeed(radPerSec), setSizeScale(scale), setPaletteFromHue(baseHue), setVisible(bool)
*/
(function(){
  const canvas = document.getElementById('bg');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  let W = 0, H = 0;

  function resize(){
    W = Math.floor(window.innerWidth);
    H = Math.floor(window.innerHeight);
    canvas.width = Math.floor(W * DPR);
    canvas.height = Math.floor(H * DPR);
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.imageSmoothingQuality = 'high';
  }
  window.addEventListener('resize', resize, { passive:true });
  resize();

  // Utilities
  const rand = (a,b)=> a + Math.random()*(b-a);
  const randInt = (a,b)=> Math.floor(rand(a,b+1));
  const clamp = (x,a,b)=> Math.max(a, Math.min(b,x));
  const lerp = (a,b,t)=> a + (b-a)*t;
  const hsl = (h,s,l,a=1)=> `hsla(${h}, ${s}%, ${l}%, ${a})`;
  const easeInOutQuad = t => t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2)/2;

  // State (headless defaults; no DOM sliders expected)
  const state = {
    rotationSpeed: 1.2,   // rad/s
    sizeScale: 1.0,
    hues: [210, 320, 40],
    targetCount: 22,
    flowers: [],
    lastTime: undefined,
    _prevDt: undefined,
  spawnTimer: 0,
  // visibility smoothing
  visibility: 0.0,           // current 0..1
  visibilityTarget: 0.0,     // target 0..1
  fadeRate: 6.0,             // per-second approach rate
  };

  function pickHue(){ const arr = state.hues; return arr[randInt(0, arr.length-1)]; }

  class Flower{
    constructor(){
      const base = Math.min(W,H);
      const spread = Math.max(W,H);
      this.x = rand(-0.1*W, 1.1*W);
      this.y = rand(-0.1*H, 1.1*H);
      this.tx = clamp(this.x + rand(-0.2*spread, 0.2*spread), -0.2*W, 1.2*W);
      this.ty = clamp(this.y + rand(-0.2*spread, 0.2*spread), -0.2*H, 1.2*H);
      this.radius = base * rand(0.03, 0.13);
      this.petals = randInt(5,12);
      this.rotation = rand(0, Math.PI*2);
      this.petalRoundness = rand(0.35, 0.75);
      this.petalFatness = rand(0.35, 0.75);
      this.doubleLayer = Math.random() < 0.35;
      this.hue = pickHue(); this.sat = rand(65,95); this.light = rand(45,65); this.targetHue = this.hue;
      this.life = 0; this.maxLife = rand(8,18);
      this.wobbleAmp = rand(0.02, 0.06) * this.radius; this.wobbleFreq = rand(0.8, 1.6);
    }
    get alpha(){
      const t = clamp(this.life/this.maxLife, 0, 1);
      const f = easeInOutQuad(t);
      const fade = Math.sin(Math.PI * f);
      return clamp(fade, 0, 1);
    }
    step(dt){
      this.life += dt;
      const t = clamp(this.life/this.maxLife, 0, 1);
      const ft = easeInOutQuad(t);
      this.x = lerp(this.x, this.tx, 0.04*dt + 0.12*ft*dt);
      this.y = lerp(this.y, this.ty, 0.04*dt + 0.12*ft*dt);
      this.rotation += state.rotationSpeed * dt;
      const hueDiff = (((this.targetHue - this.hue + 540) % 360) - 180);
      this.hue = (this.hue + hueDiff * Math.min(1, dt * 2.5) + 360) % 360;
    }
    queueHueTowardsPalette(){
      const hues = state.hues; let bestHue = this.hue; let best = Infinity;
      for(const h of hues){ const d = Math.abs((((h - this.hue + 540) % 360) - 180)); if(d < best){ best=d; bestHue=h; } }
      this.targetHue = bestHue;
    }
    draw(ctx){
      const a = this.alpha * 0.9; if(a <= 0.01) return;
      const petals = this.petals; const r = this.radius * state.sizeScale;
      ctx.save(); ctx.globalAlpha = a; ctx.translate(this.x, this.y); ctx.rotate(this.rotation);
      ctx.shadowColor = hsl(this.hue, this.sat, Math.max(50, this.light), 0.35);
      ctx.shadowBlur = r * 0.6;
      for(let i=0;i<petals;i++){
        const ang = (i/petals)*Math.PI*2; ctx.save(); ctx.rotate(ang);
        const pr = r * (i%2?1.15:0.95); const wr = pr * this.petalFatness;
        const wobble = Math.sin((performance.now()/1000)*this.wobbleFreq + i) * (this.wobbleAmp * state.sizeScale);
        const grad = ctx.createLinearGradient(0,0,pr,0);
        grad.addColorStop(0, hsl(this.hue, this.sat, Math.min(85, this.light+20), 0.95));
        grad.addColorStop(0.7, hsl(this.hue, this.sat, Math.max(15, this.light-5), 0.95));
        grad.addColorStop(1, hsl(this.hue, this.sat, Math.max(10, this.light-20), 0.92));
        ctx.fillStyle = grad; ctx.beginPath(); const round = this.petalRoundness;
        ctx.moveTo(0,0); ctx.bezierCurveTo(wr*0.25, -wr*round + wobble, pr*0.55, -wr*0.8 + wobble, pr, 0);
        ctx.bezierCurveTo(pr*0.55, wr*0.8 + wobble, wr*0.25, wr*round + wobble, 0, 0);
        ctx.closePath(); ctx.fill();
        ctx.lineWidth = Math.max(1, r*0.02); ctx.strokeStyle = hsl(this.hue, this.sat, Math.max(10, this.light-25), 0.35); ctx.stroke();
        ctx.restore();
      }
      if(this.doubleLayer){
        const innerPetals = Math.max(5, Math.round(petals*0.7)); ctx.save(); ctx.globalAlpha *= 0.9; ctx.rotate(Math.PI/innerPetals);
        for(let i=0;i<innerPetals;i++){
          const ang = (i/innerPetals)*Math.PI*2; ctx.save(); ctx.rotate(ang);
          const pr = (this.radius * state.sizeScale) * 0.65; const wr = pr * (this.petalFatness*0.8);
          const wobble = Math.sin((performance.now()/1000)*(this.wobbleFreq*1.2) + i) * (this.wobbleAmp*0.6*state.sizeScale);
          const grad = ctx.createLinearGradient(0,0,pr,0);
          grad.addColorStop(0, hsl(this.hue, this.sat, Math.min(88, this.light+28), 0.95));
          grad.addColorStop(1, hsl(this.hue, this.sat, Math.max(15, this.light-10), 0.9));
          ctx.fillStyle = grad; ctx.beginPath(); const round = Math.min(0.9, Math.max(0.2, this.petalRoundness+0.1));
          ctx.moveTo(0,0); ctx.bezierCurveTo(wr*0.25, -wr*round + wobble, pr*0.55, -wr*0.8 + wobble, pr, 0);
          ctx.bezierCurveTo(pr*0.55, wr*0.8 + wobble, wr*0.25, wr*round + wobble, 0, 0);
          ctx.closePath(); ctx.fill(); ctx.restore();
        }
        ctx.restore();
      }
      const coreR = (this.radius * state.sizeScale) * 0.2;
      const coreGrad = ctx.createRadialGradient(0,0,0,0,0,coreR);
      coreGrad.addColorStop(0, 'rgba(255,255,255,0.65)');
      coreGrad.addColorStop(1, hsl((this.hue+15)%360, 85, 28, 0.95));
      ctx.fillStyle = coreGrad; ctx.beginPath(); ctx.arc(0,0,coreR,0,Math.PI*2); ctx.fill();
      const dots = 16; const ringR = coreR*1.25;
      for(let i=0;i<dots;i++){ const ang=(i/dots)*Math.PI*2 + (this.rotation*0.2); const dx=Math.cos(ang)*ringR; const dy=Math.sin(ang)*ringR;
        ctx.beginPath(); ctx.fillStyle = hsl((this.hue+45)%360, 90, 65, 0.9); ctx.arc(dx,dy, Math.max(0.8, (this.radius*state.sizeScale)*0.03), 0, Math.PI*2); ctx.fill(); }
      ctx.restore();
    }
    get dead(){ return this.life >= this.maxLife; }
  }

  function spawnFlower(){ state.flowers.push(new Flower()); }

  function animate(ts){
    if(state.lastTime === undefined) state.lastTime = ts;
    const dtRaw = (ts - state.lastTime) / 1000;
    const dt = Math.max(0.0005, Math.min(0.03, dtRaw * 0.75 + (state._prevDt || dtRaw) * 0.25));
    state._prevDt = dt; state.lastTime = ts;
    ctx.clearRect(0,0,W,H); ctx.globalCompositeOperation='source-over';
    ctx.fillStyle = 'rgba(10,12,18,0.82)'; ctx.fillRect(0,0,W,H);
    const target = state.targetCount; state.spawnTimer -= dt;
    while(state.flowers.length < target && state.spawnTimer <= 0){ spawnFlower(); state.spawnTimer += rand(0.05, 0.15); }
    if(state.flowers.length > target){ const excess = state.flowers.length - target; for(let i=0;i<excess;i++){ const f=state.flowers[i]; f.maxLife = Math.min(f.maxLife, f.life + rand(0.6, 1.8)); } }
    for(let i=state.flowers.length-1;i>=0;i--){ const f=state.flowers[i]; f.step(dt); f.draw(ctx); if(f.dead) state.flowers.splice(i,1); }
  // Smoothly apply visibility to canvas opacity
  const k = Math.min(1, state.fadeRate * dt);
  state.visibility += (state.visibilityTarget - state.visibility) * k;
  const vis = clamp(state.visibility, 0, 1);
  canvas.style.opacity = String(vis);

  requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);

  // Public controls for gesture integration
  const control = {
    setRotationSpeed(v){ state.rotationSpeed = Math.max(0, Math.min(6.0, v)); },
    setSizeScale(s){ state.sizeScale = Math.max(0.2, Math.min(3.0, s)); },
    setPaletteFromHue(base){
      const h = ((base % 360) + 360) % 360;
      state.hues = [h, (h+40)%360, (h+300)%360];
      state.flowers.forEach(f => f.queueHueTowardsPalette());
    },
  setVisible(on){ state.visibilityTarget = on ? 1 : 0; },
  setVisibleLevel(v){ state.visibilityTarget = clamp(v, 0, 1); },
  setFadeSpeed(rate){ state.fadeRate = Math.max(0.5, Math.min(20, rate)); }
  };
  window.flowersControl = control;
})();
