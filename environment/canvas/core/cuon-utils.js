// cuon-utils.js (c) 2012 kanda and matsuda

/**

 * プログラムオブジェクトを生成し、カレントに設定する

 * @param gl GLコンテキスト

 * @param vshader 頂点シェーダのプログラム(文字列)

 * @param fshader フラグメントシェーダのプログラム(文字列)

 * @return プログラムオブジェクトを生成し、カレントの設定に成功したらtrue

 */

function initShaders(gl, vshader, fshader) {
    var program = createProgram(gl, vshader, fshader);
    if (!program) {
        console.log('failed to create program');
        return false;
    }
    gl.useProgram(program);
    gl.program = program;
    return true;
}



/**

 * リンク済みのプログラムオブジェクトを生成する

 * @param gl GLコンテキスト

 * @param vshader 頂点シェーダのプログラム(文字列)

 * @param fshader フラグメントシェーダのプログラム(文字列)

 * @return 作成したプログラムオブジェクト。作成に失敗した場合はnull

 */

function createProgram(gl, vshader, fshader) {

    // シェーダオブジェクトを作成する

    var vertexShader = loadShader(gl, gl.VERTEX_SHADER, vshader);

    var fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fshader);

    if (!vertexShader || !fragmentShader) {

        return null;

    }



    // プログラムオブジェクトを作成する

    var program = gl.createProgram();

    if (!program) {

        return null;

    }



    // シェーダオブジェクトを設定する

    gl.attachShader(program, vertexShader);

    gl.attachShader(program, fragmentShader);



    // プログラムオブジェクトをリンクする

    gl.linkProgram(program);



    // リンク結果をチェックする

    var linked = gl.getProgramParameter(program, gl.LINK_STATUS);

    if (!linked) {

        var error = gl.getProgramInfoLog(program);

        console.log('failed to link program: ' + error);

        gl.deleteProgram(program);

        gl.deleteShader(fragmentShader);

        gl.deleteShader(vertexShader);

        return null;

    }

    return program;

}



/**

 * シェーダオブジェクトを作成する

 * @param gl GLコンテキスト

 * @param type 作成するシェーダタイプ

 * @param source シェーダのプログラム(文字列)

 * @return 作成したシェーダオブジェクト。作成に失敗した場合はnull

 */

function loadShader(gl, type, source) {

    // シェーダオブジェクトを作成する

    var shader = gl.createShader(type);

    if (shader == null) {

        console.log('unable to create shader');

        return null;

    }



    // シェーダのプログラムを設定する

    gl.shaderSource(shader, source);



    // シェーダをコンパイルする

    gl.compileShader(shader);



    // コンパイル結果を検査する

    var compiled = gl.getShaderParameter(shader, gl.COMPILE_STATUS);

    if (!compiled) {

        var error = gl.getShaderInfoLog(shader);

        console.log('failed to compile shader: ' + error);

        gl.deleteShader(shader);

        return null;

    }



    return shader;

}



/**

 * attribute変数、uniform変数の位置を取得する

 * @param gl GLコンテキスト

 * @param program リンク済みのプログラムオブジェクト

 */

function loadVariableLocations(gl, program) {

    var i, name;



    // 変数の数を取得する

    var attribCount = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);

    var uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);



    // attribute変数の位置と名前の対応を取得する

    var attribIndex = {};

    for (i = 0; i < attribCount; ++i) {

        name = gl.getActiveAttrib(program, i).name;

        attribIndex[name] = i;

    }



    // uniform変数の位置と名前の対応を取得する

    var uniformLoc = {};

    for (i = 0; i < uniformCount; ++i) {

        name = gl.getActiveUniform(program, i).name;

        uniformLoc[name] = gl.getUniformLocation(program, name);

    }



    // 取得した位置をプログラムオブジェクトのプロパティとして保存する

    program.attribIndex = attribIndex;

    program.uniformLoc = uniformLoc;

}



/**

 * コンテキストを初期化して取得する

 * @param canvas 描画対象のCavnas要素

 * @param opt_debug デバッグ用の初期化をするか

 * @return 初期化を完了したGLコンテキスト

 */

function getWebGLContext(canvas, opt_debug) {

    // コンテキストを取得する

    var gl = WebGLUtils.setupWebGL(canvas);

    if (!gl) return null;



    // opt_debugに明示的にfalseが渡されなければ、デバッグ用のコンテキストを作成する

    if (arguments.length < 2 || opt_debug) {

        gl = WebGLDebugUtils.makeDebugContext(gl);

    }



    return gl;

}