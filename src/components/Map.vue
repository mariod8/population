<template>
  <div class="flex flex-col">
    <div id="canvas"></div>
  </div>
</template>
<script lang="ts">
import * as THREE from "three"
import { SVGLoader } from "three/examples/jsm/loaders/SVGLoader"
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils.js"
import CameraControls from "camera-controls"
CameraControls.install({ THREE })
import { lerp, randFloat } from "three/src/math/MathUtils"
import mapRawSvg from "@/data/map.svg"

const delta = 6
let startX: number
let startY: number

var customUVGenerator = {
  generateTopUV: function (
    geometry: THREE.ExtrudeGeometry,
    vertices: number[],
    indexA: number,
    indexB: number,
    indexC: number
  ) {
    var box = new THREE.Box3().setFromArray(vertices)
    var size = new THREE.Vector3()
    box.getSize(size)

    var a_x = vertices[indexA * 3]
    var a_y = vertices[indexA * 3 + 1]
    var b_x = vertices[indexB * 3]
    var b_y = vertices[indexB * 3 + 1]
    var c_x = vertices[indexC * 3]
    var c_y = vertices[indexC * 3 + 1]

    return [new THREE.Vector2(1, 1), new THREE.Vector2(1, 1), new THREE.Vector2(1, 1)]
  },

  generateSideWallUV: function (
    geometry: THREE.ExtrudeGeometry,
    vertices: number[],
    indexA: number,
    indexB: number,
    indexC: number,
    indexD: number
  ) {
    var box = new THREE.Box3().setFromArray(vertices)
    var size = new THREE.Vector3()
    box.getSize(size)
    var a_x = vertices[indexA * 3]
    var a_y = vertices[indexA * 3 + 1]
    var a_z = vertices[indexA * 3 + 2]
    var b_x = vertices[indexB * 3]
    var b_y = vertices[indexB * 3 + 1]
    var b_z = vertices[indexB * 3 + 2]
    var c_x = vertices[indexC * 3]
    var c_y = vertices[indexC * 3 + 1]
    var c_z = vertices[indexC * 3 + 2]
    var d_x = vertices[indexD * 3]
    var d_y = vertices[indexD * 3 + 1]
    var d_z = vertices[indexD * 3 + 2]

    return [
      new THREE.Vector2((a_x - box.min.x) / size.x, (a_z - box.min.z) / size.z),
      new THREE.Vector2((b_x - box.min.x) / size.x, (b_z - box.min.z) / size.z),
      new THREE.Vector2((c_x - box.min.x) / size.x, (c_z - box.min.z) / size.z),
      new THREE.Vector2((d_x - box.min.x) / size.x, (d_z - box.min.z) / size.z)
    ]
  }
}

export default {
  data() {
    return {
      selected: "",
      extrusion: 2
    }
  },
  methods: {
    async renderSVG() {
      const loader = new SVGLoader()
      const svgData = await loader.loadAsync(mapRawSvg)
      const svgGroup = new THREE.Group()
      var meshData: { [countryCode: string]: { geometries: THREE.BufferGeometry[] } } = {}

      svgGroup.scale.y *= -1
      svgData.paths.forEach((path) => {
        const id = path.userData?.node.id as string
        meshData[id] = {
          geometries: []
        }
      })
      var progress = 0
      svgData.paths.forEach((path) => {
        const shapes = SVGLoader.createShapes(path)

        shapes.forEach((shape) => {
          const meshGeometry = new THREE.ExtrudeGeometry(shape, {
            depth: this.extrusion,
            bevelEnabled: false,
            UVGenerator: customUVGenerator
          })
          meshData[path.userData?.node.id].geometries.push(meshGeometry)
        })
        progress += 1
        console.log(progress / svgData.paths.length)
      })
      for (const cc in meshData) {
        const geometry = BufferGeometryUtils.mergeBufferGeometries(meshData[cc].geometries)

        geometry.computeBoundingSphere()
        geometry.computeVertexNormals()
        geometry.normalizeNormals()

        const mesh = new THREE.Mesh(
          geometry,
          new THREE.ShaderMaterial({
            uniforms: {
              color1: {
                value: new THREE.Color("#222831")
              },
              color2: {
                value: new THREE.Color(
                  "#" + Math.trunc(lerp(0x24c6dc, 0x24dcca, Math.random())).toString(16)
                )
              }
            },
            vertexShader: `
    varying vec2 vUv;

    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
    }
  `,
            fragmentShader: `
    uniform vec3 color1;
    uniform vec3 color2;

    varying vec2 vUv;

    void main() {

      gl_FragColor = vec4(mix(color1, color2, vUv.y), 2.0);
    }
  `
          })
        )
        mesh.userData.id = cc
        svgGroup.add(mesh)
      }
      const box = new THREE.Box3().setFromObject(svgGroup)
      const size = box.getSize(new THREE.Vector3())
      const yOffset = size.y / -2
      const xOffset = size.x / -2

      svgGroup.children.forEach((item) => {
        item.position.x = xOffset
        item.position.y = yOffset
      })
      svgGroup.rotateX(-Math.PI / 2)
      return svgGroup
    },
    async setupScene() {
      const scene = new THREE.Scene()
      const canvas = document.getElementById("canvas")!
      const clock = new THREE.Clock()
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, depth: true })
      const pointer = new THREE.Vector2()
      const camera = new THREE.PerspectiveCamera(
        50,
        window.innerWidth / window.innerHeight,
        0.01,
        2000
      )
      camera.position.set(250, 250, 250)
      renderer.setSize(window.innerWidth, window.innerHeight)
      canvas.append(renderer.domElement)
      const ambientLight = new THREE.AmbientLight("#888888")
      const pointLight = new THREE.PointLight("#ffffff", 2, 800)
      const controls = new CameraControls(camera, renderer.domElement)
      const raycaster = new THREE.Raycaster()
      const worldMap = await this.renderSVG()

      controls.maxPolarAngle = Math.PI / 2.5
      controls.smoothTime = 0.5
      controls.maxDistance = 750

      function animate() {
        const delta = clock.getDelta()
        const updated = controls.update(delta)

        requestAnimationFrame(animate)

        if (updated) {
          renderer.render(scene, camera)
        }
      }
      scene.add(ambientLight, pointLight)
      scene.add(worldMap)
      renderer.render(scene, camera)
      animate()

      window.addEventListener("resize", () => {
        camera.aspect = window.innerWidth / window.innerHeight
        camera.updateProjectionMatrix()
        renderer.setSize(window.innerWidth, window.innerHeight)
        renderer.render(scene, camera)
      })

      window.addEventListener("mouseup", (event) => {
        const diffX = Math.abs(event.pageX - startX)
        const diffY = Math.abs(event.pageY - startY)

        if (diffX < delta && diffY < delta) {
          if (!(event.target instanceof HTMLCanvasElement)) return
          const rect = event.target.getBoundingClientRect()

          pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
          pointer.y = (-(event.clientY - rect.top) / rect.height) * 2 + 1
          raycaster.setFromCamera(pointer, camera)
          const intersects = raycaster.intersectObjects(scene.children)
          if (intersects.length) {
            const object = intersects[0].object as THREE.Mesh

            if (this.selected === object.userData.id) return
            this.selected = object.userData.id
            this.$emit("setLocation", this.selected)
            controls.fitToBox(object, true, {
              paddingLeft: 10,
              paddingRight: 10,
              paddingBottom: 10,
              paddingTop: 10
            })
            controls.rotateTo(randFloat(-Math.PI / 5, Math.PI / 5), randFloat(0.25, 0.75), true)
          }
        }
      })
      window.addEventListener("mousedown", (event) => {
        startX = event.pageX
        startY = event.pageY
      })
      controls.smoothTime = 0.5
      controls.moveTo(200, 200, 200, true)

      return scene
    }
  },
  async mounted() {
    await this.setupScene()
    this.$emit("sceneLoaded")
  }
}
</script>
