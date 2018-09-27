var UndoManager = function(canvasManager, maxStep = 20) {
	this.maxStep = maxStep;
	this.undoLog = new Array(maxStep);
	this.canvasManager = canvasManager;
	this.layerTree = canvasManager.layerTree;
	this.gl = this.layerTree.gl;
	this.undoPointer = 0;
	canvasManager.undoManager = this;
	for (var i = 0; i < maxStep; ++i)
		this.undoLog[i] = null;
};

UndoManager.prototype = {
	addStep: function(type, node, coord, posInParent = -1) {
		var step, gl = this.gl;
		if (type == 'area') {
			var width = Math.floor((coord[10] - coord[2]) * this.layerTree.canvasWidth) + 1;
			var height = Math.floor((coord[3] - coord[7]) * this.layerTree.canvasHeight) + 1;
			var x = Math.floor((1.0 + coord[0]) * node.width / 2) - 1;
			var y = Math.floor((1.0 - coord[1]) * node.height / 2) - 1;
			// refactor coord
			coord[0] = (x - node.width / 2) / (node.width / 2);
			coord[1] = (node.height / 2 - y) / (node.height / 2);
			coord[4] = coord[0];
			coord[8] = (x + width - node.width / 2) / (node.width / 2);
			coord[9] = coord[1];
			coord[5] = (node.height / 2 - y - height) / (node.height / 2);
			coord[12] = coord[8];
			coord[13] = coord[5];
			coord[2] = x / node.width;
			coord[3] = (node.height - y) / node.height;
			coord[6] = coord[2];
			coord[7] = (node.height - y - height) / node.height;
			coord[10] = (x + width) / node.width;
			coord[11] = coord[3];
			coord[14] = coord[10];
			coord[15] = coord[7];
			var c = new Float32Array(16);
			c[0] = coord[0], c[1] = coord[1], c[2] = 0.0, c[3] = 1.0,
			c[4] = coord[4], c[5] = coord[5], c[6] = 0,0, c[7] = 0.0,
			c[8] = coord[8], c[9] = coord[9], c[10] = 1.0, c[11] = 1.0,
			c[12] = coord[12], c[13] = coord[13], c[14] = 1.0, c[15] = 0.0;
			coord[0] = -1.0, coord[1] = 1.0;
			coord[4] = -1.0, coord[5] = -1.0;
			coord[8] = 1.0, coord[9] = 1.0;
			coord[12] = 1.0, coord[13] = -1.0;
			var tex = gl.createTexture();
			gl.bindTexture(gl.TEXTURE_2D, tex); 
			gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
			var fb = gl.createFramebuffer();
			gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
			gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
			gl.viewport(0, 0, width, height);
			this.layerTree.copyFromTexture(node.texture[0], coord);
			step = {type: type, data: {tex: tex, fb: fb}, coord: c, target: node};
		}
		else if (type == 'delete')
			step = {type: type, data: node, parentId: coord, pos: posInParent};
		else if (type == 'merge') {
			step = {type: type, data: node, children: coord, pos: posInParent};
			if (node.name == '*MERGE_DOWN*')
				step.tag = 'merge_down';
		}
		if (this.undoPointer >= this.maxStep) {
			if (this.undoLog[0].type == 'area') {
				gl.deleteTexture(this.undoLog[0].data.tex);
				gl.deleteFramebuffer(this.undoLog[0].data.fb);
				this.undoLog[0].data = null;
			}
			else if (this.undoLog[0].type == 'merge_down' || this.undoLog[0].type == 'merge') {
				var children = this.undoLog[0].children;
				for (var i = 0; i < children.length; ++i)
					children[i].destroy(true);
			}
			else this.undoLog[0].data.destroy(true);
			this.undoLog.shift();
			this.undoLog[this.maxStep - 1] = step;
			this.undoPointer = this.maxStep;
		}
		else {
			if (this.undoLog[this.undoPointer]) {
				if (this.undoLog[this.undoPointer].type == 'area') {
					gl.deleteTexture(this.undoLog[this.undoPointer].data.tex);
					gl.deleteFramebuffer(this.undoLog[this.undoPointer].data.fb);
					this.undoLog[this.undoPointer].data = null;
				}
				else if (this.undoLog[this.undoPointer].type == 'merge') {
					// do nothing currently
				}
			}
			this.undoLog[this.undoPointer] = step;
			++this.undoPointer;
		}
	},

	cover: function(pt) {
		var step = this.undoLog[pt];
		if (!step)
			return;
		if (step.type == 'area') {
			step.target.setActive();
			this.layerTree.copyFromTexture(step.data.tex, step.coord);
			step.target.setRenderPath(false);
		}
		else if (step.type == 'delete') {
			var parent = this.layerTree.searchNodeById(step.parentId);
			step.data.parent = parent;
			parent.children.splice(step.pos, 0, step.data);
		}
		else if (step.type == 'merge') {
			step.data.children = step.children;
			step.data.type = 'folder';
			if (step.tag)
				step.type = 'merge_down';
		}
		return step;
	},

	undo: function() {
		if (--this.undoPointer < 0) {
			this.undoPointer = 0;
			return '';
		}
		return this.cover(this.undoPointer);
	},

	redo: function() {
		if (++this.undoPointer >= this.maxStep - 1)
			this.undoPointer = this.maxStep;
		this.cover(--this.undoPointer);
	}
};