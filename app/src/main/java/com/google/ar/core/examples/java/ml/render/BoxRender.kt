package com.google.ar.core.examples.java.ml.render

import android.nfc.Tag
import android.opengl.GLES20
import android.util.Log
import com.google.ar.core.Anchor
import com.google.ar.core.PointCloud
import com.google.ar.core.Pose
import com.google.ar.core.examples.java.common.samplerender.IndexBuffer
import com.google.ar.core.examples.java.common.samplerender.Mesh
import com.google.ar.core.examples.java.common.samplerender.SampleRender
import com.google.ar.core.examples.java.common.samplerender.Shader
import com.google.ar.core.examples.java.common.samplerender.VertexBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer

class BoxRender {
    private val TAG: String = "BoxRender"
    lateinit var mesh: Mesh
    lateinit var shader: Shader

    val INDEXES = ByteBuffer.allocateDirect(1*6*4).order(
        ByteOrder.nativeOrder()
    ).asIntBuffer().apply {
        put(
            intArrayOf(0, 1, 2, 0, 2, 3)
        )
    }

    fun onSurfaceCreated(render: SampleRender) {
        GLES20.glLineWidth(10f)
        shader = Shader.createFromAssets(render,
            "shaders/box.vert",
            "shaders/box.frag",
            null)
            .setVec4(
                "u_Color", floatArrayOf(1f, 0f, 0f, 1f)
            )
    }

    fun drawRect(
        render: SampleRender,
        viewProjectionMatrix: FloatArray,
        pose: Pose,
        cameraPose: Pose,
    ){
        shader.setMat4("u_ModelViewProjection", viewProjectionMatrix)

        val centre = FloatArray(3)
        centre[0] = pose.tx()
        centre[1] = pose.ty()
        centre[2] = pose.tz()
        // get the four corners of the box
        val boxPose = Array(4) { FloatArray(3) }
        boxPose[0][0] = centre[0] - 0.5f

        val vertexBuffers = arrayOf(
            VertexBuffer(render, 3, FloatBuffer.wrap(centre)),
        )
        mesh = Mesh(
            render,
            Mesh.PrimitiveMode.LINE_LOOP,
            null,
            vertexBuffers,
        )
        render.draw(mesh, shader)
    }

    fun drawBox(
        render: SampleRender,
        viewProjectionMatrix: FloatArray,
        boxPose: Array<Pose>,
    ) {
        shader.setMat4("u_ModelViewProjection", viewProjectionMatrix)

        val topLeft = floatArrayOf(boxPose[0].tx(), boxPose[0].ty(), boxPose[0].tz())
        val bottomRight = floatArrayOf(boxPose[1].tx(), boxPose[1].ty(), boxPose[1].tz())
        val widht = bottomRight[0] - topLeft[0]
        val height = bottomRight[1] - topLeft[1]
        val z = (topLeft[2] + bottomRight[2]) / 2f

        val floatbuffer = ByteBuffer.allocateDirect(12 * 4).order(
            ByteOrder.nativeOrder()
        ).asFloatBuffer().apply {
            put(
                floatArrayOf(
                    topLeft[0], topLeft[1], z, // top left
                    topLeft[0] + widht, topLeft[1], z, // top right
                    topLeft[0] + widht, topLeft[1] + height, z, // bottom right
                    topLeft[0], topLeft[1] + height, z, // bottom left
                )
            )
        }
        val vertexBuffers = arrayOf(
            VertexBuffer(render, 3, floatbuffer),
        )
        mesh = Mesh(
            render,
            Mesh.PrimitiveMode.LINE_LOOP,
            null,
            vertexBuffers,
        )
        render.draw(mesh, shader)
    }
}