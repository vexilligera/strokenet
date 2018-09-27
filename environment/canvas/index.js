"use strict";

var defaultConfig = {
    brushSet: [
		{
			name: 'Pencil',
			opacity: 1.0,
			density: 2.0,
			polygon: 64,
			fixedColor: [-1, -1, -1, -1],
			pressureSizeSensitivity: 1.0,
			pressureColorSensitivity: 3.0,
			hotkey: 'p',
			innerThreshold: 0.5,
			blendingMode: 'NORMAL',
			enableColorMixing: false,
			mixThreshold: 0.6,
			mixStrength: 0.5,
			enableJitter: false,
			sizeJitter: 0.3,
			positionJitter: 2.0,
			tiltShading: false,
			tiltSensitivity: -1.0,
			texture: ''
		},

		{
			name: 'Brush',
			opacity: 1.0,
			density: 2.0,
			polygon: 32,
			fixedColor: [-1, -1, -1, -1],
			pressureSizeSensitivity: 1.0,
			pressureColorSensitivity: 1.0,
			hotkey: 'b',
			innerThreshold: 1.0,
			blendingMode: 'NORMAL',
			enableColorMixing: true,
			mixThreshold: 0.6,
			mixStrength: 0.5,
			enableJitter: false,
			sizeJitter: 0.3,
			positionJitter: 2.0,
			tiltShading: false,
			tiltSensitivity: -1.0,
			texture: 'http://localhost:3000/texture.png'
		},

		{
			name: 'Jitter',
			opacity: 1.0,
			density: 0.01,
			polygon: 32,
			fixedColor: [-1, -1, -1, -1],
			pressureSizeSensitivity: 1.0,
			pressureColorSensitivity: 1.0,
			hotkey: 'j',
			innerThreshold: 1.0,
			blendingMode: 'NORMAL',
			enableColorMixing: false,
			mixThreshold: 0.6,
			mixStrength: 0.5,
			enableJitter: true,
			sizeJitter: 1.0,
			positionJitter: 5.0,
			tiltShading: false,
			tiltSensitivity: -1.0,
			texture: ''
		},

		{
			name: 'Blur',
			opacity: 1.0,
			density: 0.01,
			polygon: 32,
			fixedColor: [-1, -1, -1, -1],
			pressureSizeSensitivity: 1.0,
			pressureColorSensitivity: 1.0,
			hotkey: 'f',
			innerThreshold: 1.0,
			blendingMode: 'FILTER',
			enableColorMixing: false,
			mixThreshold: 0.6,
			mixStrength: 0.5,
			enableJitter: false,
			sizeJitter: 1.0,
			positionJitter: 5.0,
			tiltShading: false,
			tiltSensitivity: -1.0,
			filterType: 'gaussian',
			kernelSize: 16,
			sigma: 20.0,
			texture: ''
		},

		{
			name: 'Eraser',
			opacity: 1.0,
			density: 0.01,
			polygon: 32,
			fixedColor: [-1, -1, -1, -1],
			pressureSizeSensitivity: 1.0,
			pressureColorSensitivity: 1.0,
			hotkey: 'e',
			innerThreshold: 1.0,
			blendingMode: 'ERASER',
			enableColorMixing: false,
			mixThreshold: 0.6,
			mixStrength: 0.5,
			enableJitter: false,
			sizeJitter: 1.0,
			positionJitter: 5.0,
			tiltShading: false,
			tiltSensitivity: -1.0,
			filterType: 'gaussian',
			kernelSize: 16,
			sigma: 20.0,
			texture: ''
		}
	],

	keyMap: {
		'undo': 'ctrl alt z',
		'thinner': '[',
		'thicker': ']',
		'magnify': 'alt =',
		'minify': 'alt -',
		'pipette': 'alt'
	},

	gesture: {
	    fingerPinch3: true,
	    fingerPinch2: true,
	    fingerRotate2: false,
	    doubleTap: true,
	    fingerRotate3: true,
	    fingerPan3Horizontal: false,
	    fingerPan3Vertical: false
    }
};

var config = new Config();
config.loadFromObject(defaultConfig);

var agent = new Object();
var canvas, layerTree, brushSet, canvasManager;

function setSize(width, height) {
	canvasManager = new CanvasManager('canvas', config.keyMap, null, width, height);
	canvas = document.getElementById('canvas');
	layerTree = new LayerTree(canvasManager, width, height, window.devicePixelRatio || 1);
	layerTree.canvasManager = canvasManager;
	brushSet = new BrushSet(layerTree);
	brushSet.loadBrushes(config.brushSet, true);
	// add white background
	var id = layerTree.createNewNode(0, layerTree.root.nodeId);
	var node = layerTree.searchNodeById(id);
	node.type = 'layer';
	node.opacity = 1.0;
	node.name = 'Background';
	node.setActive();
	node.gl.clearColor(1.0, 1.0, 1.0, 1.0);
	node.gl.clear(node.gl.COLOR_BUFFER_BIT);
	// add drawable layer
	id = layerTree.createNewNode(0, layerTree.root.nodeId);
	node = layerTree.searchNodeById(id);
	node.type = 'layer';
	node.name = 'Agent';
	node.opacity = 1.0;
	node.setActive();
	node.gl.clearColor(1.0, 1.0, 1.0, 0.0);
	node.gl.clear(node.gl.COLOR_BUFFER_BIT);
	agent.id = id;
	// use default pencil brush
	brushSet.selectBrush(0);
	return layerTree;
}

function setRadius(radius) {
	brushSet.setRadius(radius);
}

// color: normalized RGB
function setColor(color) {
	brushSet.color0 = [color[0], color[1], color[2], 1.0];
}

// points: [{x, y, pressure}, ... ]
function stroke(points) {
	brushSet.beginStroke(agent.id);
	for (var i = 0; i < points.length; ++i) {
		var pt = points[i];
		brushSet.strokeTo(pt.x, pt.y, pt.pressure);
	}
	brushSet.endStroke();
}

// fileName: image.bmp
function getImage(type) {
	layerTree.composite();
	canvasManager.updateDisplay();
	return canvas.toDataURL('image/' + type);
}

function test() {
	console.log('test api');
	setSize(1024, 1024);
	var array = [];
	for (var i = 0; i < 32; ++i) {
		array.push({
			x: Math.random() * 2 - 1,
			y: Math.random() * 2 - 1,
			pressure: Math.random()
		});
	}
	setRadius(0.1);
	setColor([0, 1, 1]);
	stroke(array);
}