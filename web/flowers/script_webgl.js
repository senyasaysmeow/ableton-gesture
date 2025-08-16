/*
  WebGL2 Flowers Renderer (GPU-accelerated)
  - Draws many soft petals as instanced quads with a fragment SDF
  - Keeps the same window.flowersControl API used by the app
  - Falls back to /static/flowers/script.js if WebGL2 is unavailable
*/
(function(){
  const canvas = document.getElementById('bg');
  if(!canvas){ return; }

  let gl = null;
  try {
    gl = canvas.getContext('webgl2', { alpha: false, antialias: true, premultipliedAlpha: false, powerPreference: 'high-performance' });
  } catch(_){ gl = null; }
  if(!gl){
    const s = document.createElement('script');
    s.src = '/static/flowers/script.js';
    document.head.appendChild(s);
    return;
  }

  // DPR-aware sizing
  const DPR = Math.max(1, Math.min(2, window.devicePixelRatio||1));
  function resize(){
    const W = Math.floor(window.innerWidth);
    const H = Math.floor(window.innerHeight);
    canvas.width = Math.floor(W * DPR);
    canvas.height = Math.floor(H * DPR);
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    gl.viewport(0,0,canvas.width, canvas.height);
  }
  window.addEventListener('resize', resize, { passive:true });
  resize();
  try { canvas.style.opacity = '1'; } catch(_){}

  // Utilities
  const clamp = (x,a,b)=> Math.max(a, Math.min(b,x));
  const rand = (a,b)=> a + Math.random()*(b-a);
  const randInt = (a,b)=> Math.floor(rand(a,b+1));
  const lerp = (a,b,t)=> a + (b-a)*t;

  // State & data
  const state = {
    rotationSpeed: 1.2,
    sizeScale: 1.0,
    hues: [210, 320, 40],
    targetCount: 22,
    flowers: [],
    lastTime: undefined,
    _prevDt: undefined,
    spawnTimer: 0,
    // visibility smoothing
    visibility: 0.0,
    visibilityTarget: 0.0,
    fadeRate: 6.0,
  };

  class Flower{
    constructor(){
      const W = canvas.width / DPR, H = canvas.height / DPR;
      const base = Math.min(W,H);
      const spread = Math.max(W,H);
      this.x = rand(-0.1*W, 1.1*W);
      this.y = rand(-0.1*H, 1.1*H);
      this.tx = clamp(this.x + rand(-0.2*spread, 0.2*spread), -0.2*W, 1.2*W);
      this.ty = clamp(this.y + rand(-0.2*spread, 0.2*spread), -0.2*H, 1.2*H);
  this.radius = base * rand(0.03, 0.13);
  this.petals = randInt(5, 12);
      this.rotation = rand(0, Math.PI*2);
      this.doubleLayer = Math.random() < 0.35;
  this.hue = state.hues[randInt(0, state.hues.length-1)];
  this.sat = rand(0.65, 0.95);
  this.light = rand(0.45, 0.65);
      this.life = 0;
      this.maxLife = rand(8, 16);
      this.wobbleAmp = rand(0.02, 0.06) * this.radius;
      this.wobbleFreq = rand(0.8, 1.6);
    }
    alpha(){
      const t = clamp(this.life / this.maxLife, 0, 1);
      const f = t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2)/2; // easeInOutQuad
      return clamp(Math.sin(Math.PI * f), 0, 1);
    }
    step(dt){
      this.life += dt;
      const t = clamp(this.life/this.maxLife, 0, 1);
      const ft = t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2)/2;
      this.x = lerp(this.x, this.tx, 0.04*dt + 0.12*ft*dt);
      this.y = lerp(this.y, this.ty, 0.04*dt + 0.12*ft*dt);
      this.rotation += state.rotationSpeed * dt;
    }
    dead(){ return this.life >= this.maxLife; }
  }

  // Shaders
  const vertSrc = `#version 300 es
  precision highp float;
  layout(location=0) in vec2 aPos;            // quad verts [-1,1]
  layout(location=1) in vec2 iTranslate;      // instance position in px
  layout(location=2) in float iSize;          // radius in px
  layout(location=3) in float iRotation;      // rotation in radians
  layout(location=4) in float iHue;           // hue 0..360
  layout(location=5) in float iAlpha;         // 0..1
  layout(location=6) in float iSat;           // 0..1
  layout(location=7) in float iLight;         // 0..1
  uniform vec2 uResolution;                   // canvas size in px
  uniform float uGlobalOpacity;               // 0..1 visibility
  out vec2 vUV;                               // [-1,1] quad coords
  out float vHue;
  out float vAlpha;
  out float vSat;
  out float vLight;
  void main(){
    // rotate quad in NDC by iRotation
    float s = sin(iRotation), c = cos(iRotation);
    vec2 p = vec2(c*aPos.x - s*aPos.y, s*aPos.x + c*aPos.y);
    // scale to px and translate
    vec2 world = iTranslate + p * iSize;
    // to NDC
    vec2 ndc = (world / uResolution) * 2.0 - 1.0;
    ndc.y *= -1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUV = aPos; // use for radial SDF
    vHue = iHue;
    vAlpha = iAlpha * uGlobalOpacity;
    vSat = iSat;
    vLight = iLight;
  }
  `;

  const fragSrc = `#version 300 es
  precision highp float;
  in vec2 vUV; // [-1,1]
  in float vHue;
  in float vAlpha;
  in float vSat;
  in float vLight;
  out vec4 outColor;
  uniform int uMode; // 0=petal, 1=core, 2=dot, 3=shadow

  // Simple HSL->RGB
  vec3 hsl2rgb(float h, float s, float l){
    float c = (1.0 - abs(2.0*l - 1.0)) * s;
    float hp = h/60.0;
    float x = c * (1.0 - abs(mod(hp, 2.0) - 1.0));
    vec3 rgb;
    if(hp < 1.0) rgb = vec3(c,x,0); else if(hp < 2.0) rgb = vec3(x,c,0);
    else if(hp < 3.0) rgb = vec3(0,c,x); else if(hp < 4.0) rgb = vec3(0,x,c);
    else if(hp < 5.0) rgb = vec3(x,0,c); else rgb = vec3(c,0,x);
    float m = l - 0.5*c; return rgb + m;
  }

  void main(){
    vec2 uv = vUV; // [-1,1]
    if(uMode == 0){
      // Petal SDF: super-ellipse-like teardrop with soft edge
      // Taper petals towards the tip (uv.x ~ +1.0)
      float t = clamp(uv.x*0.5 + 0.5, 0.0, 1.0);
      float taperY = mix(1.0, 0.65, t);
      vec2 uvp = vec2(uv.x, uv.y / taperY);
      float rx = 1.0; float ry = 0.58;
      float d = pow(abs(uvp.x)/rx, 2.0) + pow(abs(uvp.y)/ry, 2.1); // superellipse
  float body = smoothstep(1.0, 0.84, d);
  // Gentle dark rim
  float rim = smoothstep(0.94, 1.0, d);
      // Gradient along axis
      float g = clamp(uv.x*0.5 + 0.5, 0.0, 1.0);
      float hue = mod(vHue, 360.0);
  float s = clamp(vSat * 1.02, 0.0, 1.0);
      // brighten towards tip, darker at base
      float l0 = clamp(vLight + 0.20, 0.0, 1.0);
      float l1 = clamp(vLight - 0.20, 0.0, 1.0);
      vec3 c0 = hsl2rgb(hue, s, l0);
      vec3 c1 = hsl2rgb(hue, s, l1);
      vec3 col = mix(c0, c1, g);
      // Apply rim darkening
  col *= (1.0 - rim*0.28);
      float alpha = body * vAlpha;
      if(alpha < 0.01) discard;
      outColor = vec4(col, alpha);
    } else if(uMode == 1){
      // Core circle with radial glow (white center to darker hue)
      float r = length(uv);
      float edge = smoothstep(1.0, 0.0, r);
      float lCenter = clamp(vLight + 0.28, 0.0, 1.0);
      float lEdge = clamp(vLight - 0.25, 0.0, 1.0);
      vec3 center = vec3(1.0);
  vec3 edgeCol = hsl2rgb(mod(vHue+15.0,360.0), clamp(vSat*1.02,0.0,1.0), lEdge);
      vec3 col = mix(center, edgeCol, smoothstep(0.0,1.0,r));
      float alpha = edge * vAlpha;
      if(alpha < 0.01) discard;
      outColor = vec4(col, alpha);
    } else if(uMode == 2){
      // Ring dots: small bright circles
  float r = length(uv);
      float edge = smoothstep(1.0, 0.0, r);
  vec3 col = hsl2rgb(mod(vHue+45.0,360.0), 0.95, 0.68);
      float alpha = edge * vAlpha;
      if(alpha < 0.01) discard;
      outColor = vec4(col, alpha);
    } else {
      // Shadow glow: dark radial soft circle
  float r = length(uv);
  float edge = smoothstep(1.0, 0.0, r);
  vec3 col = hsl2rgb(vHue, clamp(vSat,0.0,1.0), 0.18);
  float alpha = edge * vAlpha * 0.25;
      if(alpha < 0.01) discard;
      outColor = vec4(col, alpha);
    }
  }
  `;

  function compile(type, src){
    const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
    if(!gl.getShaderParameter(sh, gl.COMPILE_STATUS)){
      console.error('Shader compile error:', gl.getShaderInfoLog(sh));
      gl.deleteShader(sh); return null;
    }
    return sh;
  }
  function link(vs, fs){
    const prog = gl.createProgram(); gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    if(!gl.getProgramParameter(prog, gl.LINK_STATUS)){
      console.error('Program link error:', gl.getProgramInfoLog(prog));
      gl.deleteProgram(prog); return null;
    }
    return prog;
  }

  const vs = compile(gl.VERTEX_SHADER, vertSrc);
  const fs = compile(gl.FRAGMENT_SHADER, fragSrc);
  if(!vs || !fs){
    const s = document.createElement('script'); s.src = '/static/flowers/script.js'; document.head.appendChild(s); return;
  }
  const prog = link(vs, fs);
  gl.useProgram(prog);

  // Geometry: unit quad [-1,1] in both axes
  const quad = new Float32Array([
    -1,-1,  1,-1, -1, 1,
     1,-1,  1, 1, -1, 1
  ]);
  const vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vbo); gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  // Instance buffer layout
  const maxInstances = 8192;
  const stride = (2+1+1+1+1+1+1) * 4; // vec2 + size + rot + hue + alpha + sat + light
  const instanceBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
  gl.bufferData(gl.ARRAY_BUFFER, maxInstances * stride, gl.DYNAMIC_DRAW);

  let offset = 0;
  // iTranslate (vec2)
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, stride, offset); gl.vertexAttribDivisor(1, 1);
  offset += 2*4;
  // iSize (float)
  gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, stride, offset); gl.vertexAttribDivisor(2, 1);
  offset += 4;
  // iRotation (float)
  gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 1, gl.FLOAT, false, stride, offset); gl.vertexAttribDivisor(3, 1);
  offset += 4;
  // iHue (float)
  gl.enableVertexAttribArray(4); gl.vertexAttribPointer(4, 1, gl.FLOAT, false, stride, offset); gl.vertexAttribDivisor(4, 1);
  offset += 4;
  // iAlpha (float)
  gl.enableVertexAttribArray(5); gl.vertexAttribPointer(5, 1, gl.FLOAT, false, stride, offset); gl.vertexAttribDivisor(5, 1);
  offset += 4;
  // iSat (float)
  gl.enableVertexAttribArray(6); gl.vertexAttribPointer(6, 1, gl.FLOAT, false, stride, offset); gl.vertexAttribDivisor(6, 1);
  offset += 4;
  // iLight (float)
  gl.enableVertexAttribArray(7); gl.vertexAttribPointer(7, 1, gl.FLOAT, false, stride, offset); gl.vertexAttribDivisor(7, 1);

  const uResolution = gl.getUniformLocation(prog, 'uResolution');
  const uGlobalOpacity = gl.getUniformLocation(prog, 'uGlobalOpacity');
  const uMode = gl.getUniformLocation(prog, 'uMode');

  gl.enable(gl.BLEND);
  // Straight alpha on an opaque canvas; this avoids double-premultiplication gray artifacts in Arc
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  function spawnFlower(){ state.flowers.push(new Flower()); }

  function buildPetalInstances(){
    // Prepare instance data for all petals
    const instances = [];
    const twoPI = Math.PI*2;
    for(const f of state.flowers){
      const alpha = f.alpha();
      const petals = f.petals;
      const layerCount = f.doubleLayer ? 2 : 1;
      const sat = f.sat; // per-flower saturation
      const light = f.light; // per-flower base lightness
      for(let layer=0; layer<layerCount; layer++){
        const petCount = layer==0 ? petals : Math.max(5, Math.round(petals*0.7));
        const baseR = (layer==0 ? f.radius : f.radius*0.65) * state.sizeScale;
        for(let i=0;i<petCount;i++){
          const ang = (i/petCount)*twoPI + (layer==1 ? Math.PI/petCount : 0);
          const rot = f.rotation + ang;
          const hue = f.hue;
          const size = baseR * (i % 2 ? 1.15 : 0.95); // alternate lengths like 2D
          // translate.x, translate.y, size, rotation, hue, alpha, sat, light
          instances.push(f.x, f.y, size, rot, hue, alpha, sat, light);
        }
      }
    }
    return new Float32Array(instances);
  }

  function buildCoreInstances(){
    const instances = [];
    for(const f of state.flowers){
      const alpha = f.alpha();
      const coreR = (f.radius * state.sizeScale) * 0.2;
      const hue = f.hue; const sat = Math.min(1.0, f.sat + 0.10); const light = Math.max(0.0, f.light - 0.10);
      instances.push(f.x, f.y, coreR, 0.0, hue, alpha, sat, light);
    }
    return new Float32Array(instances);
  }

  function buildDotInstances(){
    const instances = [];
    const dots = 16;
    for(const f of state.flowers){
      const alpha = f.alpha()*0.9;
      const ringR = (f.radius * state.sizeScale) * 0.25; // 1.25 * coreR (coreR=0.2*radius)
      const hue = f.hue; const sat = Math.min(1.0, f.sat + 0.2); const light = Math.min(1.0, f.light + 0.1);
      for(let i=0;i<dots;i++){
        const ang = (i/dots)*Math.PI*2 + (f.rotation*0.2);
        const dx = Math.cos(ang) * ringR;
        const dy = Math.sin(ang) * ringR;
        const size = Math.max(0.8, (f.radius*state.sizeScale)*0.03);
        instances.push(f.x+dx, f.y+dy, size, 0.0, hue, alpha, sat, light);
      }
    }
    return new Float32Array(instances);
  }

  function buildShadowInstances(){
    const instances = [];
    for(const f of state.flowers){
      const alpha = f.alpha();
  const size = (f.radius * state.sizeScale) * 0.82; // reduce to avoid circular blob
      const hue = f.hue; const sat = Math.max(0.0, f.sat - 0.2); const light = 0.20;
      instances.push(f.x, f.y, size, 0.0, hue, alpha, sat, light);
    }
    return new Float32Array(instances);
  }

  function animate(ts){
    if(state.lastTime===undefined) state.lastTime = ts;
    const dtRaw = (ts - state.lastTime)/1000;
    const dt = Math.max(0.0005, Math.min(0.03, dtRaw*0.75 + (state._prevDt||dtRaw)*0.25));
    state._prevDt = dt; state.lastTime = ts;

  // background: fade darkness with visibility while keeping opaque alpha
  const vis = clamp(state.visibility, 0, 1);
  const bgScale = 0.15 + 0.85 * vis; // 0.15 when hidden, 1.0 when fully visible
  gl.clearColor((10/255)*bgScale, (12/255)*bgScale, (18/255)*bgScale, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);

    // spawn/retire
    state.spawnTimer -= dt;
    while(state.flowers.length < state.targetCount && state.spawnTimer <= 0){ spawnFlower(); state.spawnTimer += rand(0.05, 0.15); }
    if(state.flowers.length > state.targetCount){
      const excess = state.flowers.length - state.targetCount;
      for(let i=0;i<excess;i++){
        const f = state.flowers[i]; f.maxLife = Math.min(f.maxLife, f.life + rand(0.6, 1.8));
      }
    }

    for(let i=state.flowers.length-1;i>=0;i--){
      const f = state.flowers[i]; f.step(dt); if(f.dead()) state.flowers.splice(i,1);
    }

    // visibility smoothing
  const k = Math.min(1, state.fadeRate * dt);
  state.visibility += (state.visibilityTarget - state.visibility) * k;

  // Upload and draw: shadow (behind petals)
  let arr = buildShadowInstances();
  let instanceCount = Math.min(arr.length / 8, maxInstances);
  gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, arr.subarray(0, instanceCount*8));

  gl.useProgram(prog);
  gl.uniform2f(uResolution, canvas.width / DPR, canvas.height / DPR);
  gl.uniform1f(uGlobalOpacity, vis);
  gl.uniform1i(uMode, 3);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, instanceCount);

  // Upload and draw: petals
  arr = buildPetalInstances();
  instanceCount = Math.min(arr.length / 8, maxInstances);
  gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, arr.subarray(0, instanceCount*8));

  // uniforms common
  gl.useProgram(prog);
  gl.uniform2f(uResolution, canvas.width / DPR, canvas.height / DPR);
  gl.uniform1f(uGlobalOpacity, vis);

  gl.uniform1i(uMode, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, instanceCount);

  // Core pass
  arr = buildCoreInstances();
  instanceCount = Math.min(arr.length / 8, maxInstances);
  gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, arr.subarray(0, instanceCount*8));
  gl.uniform1i(uMode, 1);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, instanceCount);

  // Dots pass
  arr = buildDotInstances();
  instanceCount = Math.min(arr.length / 8, maxInstances);
  gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, arr.subarray(0, instanceCount*8));
  gl.uniform1i(uMode, 2);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, instanceCount);

    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);

  // Controls exposed to window
  const control = {
    setRotationSpeed(v){ state.rotationSpeed = Math.max(0, Math.min(6.0, v)); },
    setSizeScale(s){ state.sizeScale = Math.max(0.2, Math.min(3.0, s)); },
    setPaletteFromHue(base){
      const h = ((base % 360) + 360) % 360; state.hues = [h, (h+40)%360, (h+300)%360];
      // Retint existing flowers gradually (towards nearest palette hue)
      for(const f of state.flowers){
        let best = f.hue, dMin = 1e9;
        for(const ph of state.hues){ const d = Math.abs((((ph - f.hue + 540) % 360) - 180)); if(d < dMin){ dMin = d; best = ph; } }
        f.hue = (f.hue*0.9 + best*0.1);
      }
    },
    setVisible(on){ state.visibilityTarget = on ? 1 : 0; },
    setVisibleLevel(v){ state.visibilityTarget = clamp(v, 0, 1); },
    setFadeSpeed(rate){ state.fadeRate = Math.max(0.5, Math.min(20, rate)); },
  };
  window.flowersControl = control;
})();
