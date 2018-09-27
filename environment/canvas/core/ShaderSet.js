var ShaderSet = function(gl) {
	this.blendShaderCount = 3;
	this.blendVertexShaders = new Array(this.blendShaderCount);
	this.blendFragmentShaders = new Array(this.blendShaderCount);
	this.blendShaderPrograms = new Array(this.blendShaderCount);

	this.brushVertexShaders = new Array(1);
	this.brushFragmentShaders = new Array(1);
	this.brushShaderPrograms = new Array(1);
	this.brushShaderCount = 1;

	this.filterShaderCount = 2;
	this.filterVertexShaders = new Array(this.filterShaderCount);
	this.filterFragmentShaders = new Array(this.filterShaderCount);
	this.filterShaderPrograms = new Array(this.filterShaderCount);

	this.utilShadersCount = 1;
	this.utilShaderPrograms = new Array(this.utilShadersCount);
	this.utilVertexShaders = new Array(this.utilShadersCount);
	this.utilFragmentShaders = new Array(this.utilShadersCount);

	// alpha blend
	this.blendVertexShaders[0] =
		'attribute vec4 a_Position;' +
		'attribute vec2 a_TexCoord;' +
		'varying vec2 v_TexCoord;' +
		'void main() {' +
		'	gl_Position = a_Position;' +
		'	v_TexCoord = a_TexCoord;' +
		'}';
	this.blendFragmentShaders[0] =
		'precision mediump float;' +
		'uniform sampler2D u_CanvasSampler;' +
		'uniform sampler2D u_ChildSampler;' +
		'uniform float u_CanvasOpacity;' +
		'uniform float u_ChildOpacity;' +
		'varying vec2 v_TexCoord;' +
		'void main() {' +
		'	vec4 colorChild = texture2D(u_ChildSampler, v_TexCoord);' +
		'	vec4 colorBase = texture2D(u_CanvasSampler, v_TexCoord);' +
		'	colorChild.w = colorChild.w * u_ChildOpacity;' +
		'	colorBase.w = colorBase.w * u_CanvasOpacity;' +
		'	float alpha = colorChild.w + colorBase.w - colorChild.w * colorBase.w;' +
		'	gl_FragColor = (colorChild * colorChild.w + colorBase * colorBase.w * (1.0 - colorChild.w))/alpha;' +
		'	gl_FragColor.w = alpha;' +
		'}';

	// multiply blend
	this.blendVertexShaders[1] = this.blendVertexShaders[0];
	this.blendFragmentShaders[1] = 
		'precision mediump float;' +
		'uniform sampler2D u_CanvasSampler;' +
		'uniform sampler2D u_ChildSampler;' +
		'uniform float u_CanvasOpacity;' +
		'uniform float u_ChildOpacity;' +
		'varying vec2 v_TexCoord;' +
		'void main() {' +
		'	vec4 colorChild = texture2D(u_ChildSampler, v_TexCoord);' +
		'	vec4 colorBase = texture2D(u_CanvasSampler, v_TexCoord);' +
		'	colorChild.w = colorChild.w * u_ChildOpacity;' +
		'	colorBase.w = colorBase.w * u_CanvasOpacity;' +
		'	float alpha = colorChild.w + colorBase.w - colorChild.w * colorBase.w;' +
		'	gl_FragColor = colorChild * colorBase;' +
		//'	gl_FragColor.w = alpha;' +
		'}';

	// stencil
	this.blendVertexShaders[2] = this.blendVertexShaders[0];
	this.blendFragmentShaders[2] =
		'precision mediump float;' +
		'uniform sampler2D u_CanvasSampler;' +
		'uniform sampler2D u_ChildSampler;' +
		'uniform float u_CanvasOpacity;' +
		'uniform float u_ChildOpacity;' +
		'varying vec2 v_TexCoord;' +
		'void main() {' +
		'	vec4 colorChild = texture2D(u_ChildSampler, v_TexCoord);' +
		'	vec4 colorBase = texture2D(u_CanvasSampler, v_TexCoord);' +
		'	gl_FragColor = colorBase;' +
		'	gl_FragColor.w *= 1.0 - colorChild.w;' +
		'}';

	// General brush shader
	this.brushVertexShaders[0] =
		'precision mediump float;' +
		'varying vec4 v_CenterPos;' +
		'varying float v_Radius;' +
		'varying float v_Pressure;' +
		'varying vec2 v_Tilt;' +
		'varying vec4 v_Color;' +
		'varying float v_Time;' +
		'varying vec2 v_TexCoord;' +
		'varying vec4 v_Position;' +
		'varying vec4 avg_clr;' +
		'attribute vec4 a_Position;' +
		'attribute vec4 a_CenterPos;' +
		'attribute float a_Radius;' +
		'attribute float a_Pressure;' +
		'attribute vec2 a_Tilt;' +
		'attribute vec4 a_Color;' +
		'attribute float a_Time;' +
		'uniform sampler2D u_brushTexture;' +
		'uniform float u_mixThreshold;' +
		'uniform vec2 u_xyFactor;' +
		'uniform float u_tiltSensitivity;' +
		'uniform sampler2D u_layerSampler;' +
		'uniform float u_pressureColorSensitivity;' +
		'void main() {' +
		'	gl_Position = a_Position;' +
		'	float pressure = a_Pressure;' +
		'	if (u_mixThreshold != -1.0) {' +
		'		float t = sqrt(2.0) / 2.0 * a_Radius;' +
		'		vec2 pos = a_Position.xy - (t, t);' +
		'		vec2 texpos;' +
		'		float initx = pos.x;' +
		'		vec2 step = vec2(2.0 * t, 2.0 * t) * u_xyFactor / 10.0;' +
		'		avg_clr = vec4(0.0, 0.0, 0.0, 0.0);' +
		'		for (int i = 0; i < 5; ++i) {' +
		'			for (int j = 0; j < 5; ++j) {' +
		'				texpos = (pos + vec2(1.0, 1.0)) / 2.0;' +
		'				avg_clr = avg_clr + texture2D(u_layerSampler, texpos);' +
		'				pos.x = pos.x + step.x;' +
		'			}' +
		'			pos.x = initx;' +
		'			pos.y = pos.y + step.y;' +
		'		}' +
		'		avg_clr /= 25.0;' +
		'	}' +
		'	if (u_tiltSensitivity != -1.0) {' +
		'		vec2 v1 = (a_Position - v_CenterPos).xy;' +
		'		float _cos = dot(v1, a_Tilt) / (length(v1) * length(a_Tilt));' +
		'		pressure = (1.0 - 0.5 * _cos) * a_Pressure;' +
		'	}' +
		'	v_Position = a_Position;' +
		'	v_CenterPos = a_CenterPos;' +
		'	v_Radius = a_Radius;' +
		'	v_Pressure = a_Color[3] * pow(pressure, u_pressureColorSensitivity);' +
		'	v_Tilt = a_Tilt;' +
		'	v_Color = a_Color;' +
		'	v_Time = a_Time;' +
		'	v_TexCoord.x = (a_Position.x + 1.0) / 2.0;' +
		'	v_TexCoord.y = (a_Position.y + 1.0) / 2.0;' +
		'}';
	this.brushFragmentShaders[0] = 
		'precision mediump float;' +
		'uniform sampler2D u_layerSampler;' +
		'uniform sampler2D u_brushTexture;' +
		'uniform vec2 u_xyFactor;' +
		'uniform float u_InnerThreshold;' +
		'uniform int u_brushHasTexture;' +
		'uniform float u_mixThreshold;' +
		'uniform float u_mixStrength;' +
		'uniform float u_pressureColorSensitivity;' +
		'varying vec2 v_TexCoord;' +
		'varying vec4 v_Position;' +
		'varying vec4 v_CenterPos;' +
		'varying float v_Radius;' +
		'varying float v_Pressure;' +
		'varying vec2 v_Tilt;' +
		'varying vec4 v_Color;' +
		'varying float v_Time;' +
		'varying vec4 avg_clr;' +
		'void main() {' +
		'	vec4 base_color = texture2D(u_layerSampler, v_TexCoord);' +
		'	float d = length((v_CenterPos.xy - v_Position.xy) * u_xyFactor.yx);' +
		'	float r = d / v_Radius;' +
		'	float e = 1.0 - r;' +
		'	float factor = 1.0 / u_InnerThreshold;' +
		'	float opacity;' +
		'	float pressure = v_Pressure;' +
		'	vec4 color = v_Color;' +
		'	if (e < u_InnerThreshold) {' +
		'		pressure *= e * factor;' +
		'	}' +
		'	opacity = pressure;' +
		'	if (u_mixThreshold != -1.0) {' +
		'		float s = u_mixThreshold / r;' +
		'		float t = v_Pressure * (1.0 - avg_clr.w);' +
		'		float d = 1.0 - u_mixStrength * avg_clr.w;' +
		'		if (d == 0.0) t = 1.0;' +
		'		else t = sqrt(t) / (1.0 / d) + d;' +
		//'		s = s / 20.0 + 0.95;' +
		'		if (s > 0.1)' +
		'			color = mix(avg_clr, color, t);' +
		'		else' +
		'			color = mix(avg_clr, color, s * t);' +
		'	}' +
		'	color.w = opacity;' +
		'	if (u_brushHasTexture == 1) {' +
		'		vec2 texpos = ((1.0 + (v_Position - v_CenterPos) / v_Radius) / 2.0).xy;' +
		'		vec4 texcolor = texture2D(u_brushTexture, texpos);' +
		'		color.w = color.w * texcolor.w;' +
		'	}' +
		'	gl_FragColor = color;' +
		'}'; 

	// Photoshop No. 19 Brush
	this.brushVertexShaders[1] =
		'varying vec4 v_CenterPos;' +
		'varying float v_Radius;' +
		'varying float v_Pressure;' +
		'varying vec2 v_Tilt;' +
		'varying vec4 v_Color;' +
		'varying float v_Time;' +
		'varying vec2 v_TexCoord;' +
		'varying vec4 v_Position;' +
		'attribute vec4 a_Position;' +
		'attribute vec4 a_CenterPos;' +
		'attribute float a_Radius;' +
		'attribute float a_Pressure;' +
		'attribute vec2 a_Tilt;' +
		'attribute vec4 a_Color;' +
		'attribute float a_Time;' +
		'void main() {' +
		'	gl_Position = a_Position;' +
		'	v_Position = a_Position;' +
		'	v_CenterPos = a_CenterPos;' +
		'	v_Radius = a_Radius;' +
		'	v_Pressure = a_Pressure;' +
		'	v_Tilt = a_Tilt;' +
		'	v_Color = a_Color;' +
		'	v_Time = a_Time;' +
		'	v_TexCoord.x = (a_Position.x + 1.0) / 2.0;' +
		'	v_TexCoord.y = (a_Position.y + 1.0) / 2.0;' +
		'}';
	this.brushFragmentShaders[1] = 
		'precision mediump float;' +
		'uniform sampler2D u_layerSampler;' +
		'uniform vec2 u_xyFactor;' +
		'uniform float u_Threshold;' +
		'varying vec2 v_TexCoord;' +
		'varying vec4 v_Position;' +
		'varying vec4 v_CenterPos;' +
		'varying float v_Radius;' +
		'varying float v_Pressure;' +
		'varying vec2 v_Tilt;' +
		'varying vec4 v_Color;' +
		'varying float v_Time;' +
		'void main() {' +
		'	vec4 colorLayer = texture2D(u_layerSampler, v_TexCoord);' +
		'	float r = length((v_CenterPos.xy - v_Position.xy) * u_xyFactor.yx) / v_Radius;' +
		'	float e = 1.0 - r;' +
		'	float factor = 1.0 / u_Threshold;' +
		'	float opacity;' +
		'	float pressure = v_Pressure * v_Pressure;' +
		'	if (e > u_Threshold) {' +
		'		opacity = pressure + colorLayer.w - pressure * colorLayer.w;' +
		'		gl_FragColor = (v_Color * v_Color.w * opacity + colorLayer * colorLayer.w * (1.0 - v_Color.w * opacity))/opacity;' +
		'		gl_FragColor.w = opacity;' +
		'	}' +
		'	else {' +
		'		opacity = pressure * e * factor;' +
		'		gl_FragColor = v_Color;' +
		'		gl_FragColor.w = opacity;' +
		'	}' +
		'}'; 

	// plain copy filter
	this.filterVertexShaders[0] = 
		'attribute vec4 a_Position;' +
		'attribute vec2 a_TexCoord;' +
		'varying vec2 v_TexCoord;' +
		'void main() {' +
		'	gl_Position = a_Position;' +
		'	v_TexCoord = a_TexCoord;' +
		'}';
	this.filterFragmentShaders[0] =
		'precision mediump float;' +
		'uniform sampler2D u_Sampler;' +
		'varying vec2 v_TexCoord;' +
		'void main() {' +
		'	gl_FragColor = texture2D(u_Sampler, v_TexCoord);' +
		'}';

	// warping filter
	this.filterVertexShaders[1] =
		'varying vec2 v_texCoord;' +
		'varying vec2 v_vertexCoord;' +
		'attribute vec2 a_texCoord;' +
		'attribute vec2 a_vertexCoord;' +
		'void main() {' +
		'	v_texCoord = a_texCoord;' +
		'	v_vertexCoord = a_vertexCoord;' +
		'	gl_Position = vec4(a_vertexCoord, 0.0, 1.0);' +
		'}';
	this.filterFragmentShaders[1] =
		'precision mediump float;' +
		'uniform sampler2D u_texture;' +
		'varying vec2 v_texCoord;' +
		'void main() {' +
		'	gl_FragColor = texture2D(u_texture, v_texCoord);' +
		'}';

	// gaussian blur
	this.filterVertexShaders[2] =
		'attribute vec2 a_Position;' +
		'attribute float a_Pressure;' +
		'varying float v_Pressure;' +
		'varying vec2 v_Position;' +
		'uniform float u_kernel[256];' +
		'void main() {' +
		'	gl_Position = vec4(a_Position.x, a_Position.y, 0.0, 1.0);' +
		'	v_Position = a_Position;' +
		'	v_Pressure = a_Pressure;' +
		'}';
	this.filterFragmentShaders[2] =
		'precision highp float;' +
		'varying vec2 v_Position;' +
		'varying float v_Pressure;' +
		'uniform vec2 u_xyStep;' +
		'uniform float u_kernelSize;' +
		'uniform sampler2D u_layerSampler;' +
		'uniform float u_kernel[256];' +
		'void main() {' +
		'	vec4 color;' +
		'	vec2 step = u_xyStep / 2.0;' +
		'	vec2 coord = (v_Position + 1.0) / 2.0;' +
		'	coord = coord - float(int(u_kernelSize / 2.0)) * step;' +
		'	float initx = coord.x;' +
		'	vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);' +
		'	int size = int(u_kernelSize * u_kernelSize);' +
		'	int kernelSize = int(u_kernelSize);' +
		'	int c = 0;' +
		'	for (int i = 0; i < 256; ++i) {' +
		'		if (i >= size) break;' +
		'		color = texture2D(u_layerSampler, coord);' +
		'		if (color.w == 0.0) color = vec4(1.0, 1.0, 1.0, 0.0);' +
		'		sum += u_kernel[i] * color;' +
		'		coord.x += step.x;' +
		'		++c;' +
		'		if (c == kernelSize) {' +
		'			c = 0;' +
		'			coord.y += step.y;' +
		'			coord.x = initx;' +
		'		}' +
		'	}' +
		'	gl_FragColor = sum;' +
		'}';

	this.utilVertexShaders[0] =
		'attribute vec4 a_Position;' +
		'void main() {' +
		'	gl_Position = a_Position;' +
		'}';
	this.utilFragmentShaders[0] =
		'precision mediump float;' +
		'uniform vec4 u_Color;' +
		'void main() {' +
		'	gl_FragColor = u_Color;' +
		'}';

	this.gl = gl;
	this.typeBlendShader = 'blend';
	this.typeBrushShader = 'brush';
	this.typeFilterShader = 'filter';
	this.typeUtilShader = 'util';

	for (var i = 0; i < this.blendShaderCount; ++i)
		this.blendShaderPrograms[i] = null;
	for (var i = 0; i < this.brushShaderCount; ++i)
		this.brushShaderPrograms[i] = null;
	for (var i = 0; i < this.filterShaderCount; ++i)
		this.filterShaderPrograms[i] = null;
	for (var i = 0; i < this.utilShadersCount; ++i)
		this.utilShaderPrograms[i] = null;
	this.compileAllShaders();
}

ShaderSet.prototype = {
	compileShader: function(vshader, fshader) {
		var program = createProgram(this.gl, vshader, fshader);
		if (!program) {
			// TODO: signal error to the error handling system
			return null;
		}
		return program;
	},

	selectShader: function(type, i) {
		var shaders, vs, fs;
		if (type == this.typeBlendShader) {
			shaders = this.blendShaderPrograms;
			vs = this.blendVertexShaders;
			fs = this.blendFragmentShaders;
		}
		else if (type == this.typeBrushShader) {
			shaders = this.brushShaderPrograms;
			vs = this.brushVertexShaders;
			fs = this.brushFragmentShaders;
		}
		else if (type == this.typeFilterShader) {
			shaders = this.filterShaderPrograms;
			vs = this.filterVertexShaders;
			fs = this.filterFragmentShaders;
		}
		else if (type == this.typeUtilShader) {
			shaders = this.utilShaderPrograms;
			vs = this.utilVertexShaders;
			fs = this.utilFragmentShaders;
		}
		else
			return false;
		if (shaders[i] == null) {
			shaders[i] = this.compileShader(vs[i], fs[i]);
			console.log(shaders,i);
		}
		this.gl.useProgram(shaders[i]);
		this.gl.program = shaders[i];
		return true;
	},

	compileAllShaders: function() {
		for (var i = 0; i < this.blendShaderCount; ++i)
			this.blendShaderPrograms[i] = this.compileShader(this.blendVertexShaders[i], this.blendFragmentShaders[i]);
		for (var i = 0; i < this.brushShaderCount; ++i)
			this.brushShaderPrograms[i] = this.compileShader(this.brushVertexShaders[i], this.brushFragmentShaders[i]);
		for (var i = 0; i < this.filterShaderCount; ++i)
			this.filterShaderPrograms[i] = this.compileShader(this.filterVertexShaders[i], this.filterFragmentShaders[i]);
		for (var i = 0; i < this.utilShadersCount; ++i)
			this.utilShaderPrograms[i] = this.compileShader(this.utilVertexShaders[i], this.utilFragmentShaders[i]);
	}
};