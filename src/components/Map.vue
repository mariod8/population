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
import { randomHexColor } from "@/utils"
import { randFloat } from "three/src/math/MathUtils"

export default {
  data() {
    return {
      selected: "",
      extrusion: 5
    }
  },
  methods: {
    async renderSVG() {
      const loader = new SVGLoader()
      const svgData = await loader.loadAsync("src/data/map.svg")
      const svgGroup = new THREE.Group()
      var meshData: { [countryCode: string]: { geometries: THREE.BufferGeometry[] } } = {}

      svgGroup.scale.y *= -1
      svgData.paths.forEach((path) => {
        const id = path.userData?.node.id as string
        meshData[id] = {
          geometries: []
        }
      })

      svgData.paths.forEach((path) => {
        const shapes = SVGLoader.createShapes(path)

        shapes.forEach((shape) => {
          const meshGeometry = new THREE.ExtrudeGeometry(shape, {
            depth: this.extrusion,
            bevelEnabled: false
          })
          meshData[path.userData?.node.id].geometries.push(meshGeometry)
        })
      })
      for (const cc in meshData) {
        const geometry = BufferGeometryUtils.mergeBufferGeometries(meshData[cc].geometries)

        geometry.computeBoundingSphere()
        geometry.computeVertexNormals()
        geometry.normalizeNormals()

        const mesh = new THREE.Mesh(
          geometry,
          new THREE.MeshBasicMaterial({ color: randomHexColor() })
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

      window.addEventListener("click", (event) => {
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
          this.$emit('setLocation', this.selected)
          controls.fitToBox(object, true, {
            paddingLeft: 10,
            paddingRight: 10,
            paddingBottom: 10,
            paddingTop: 10
          })
          controls.rotateTo(randFloat(-Math.PI / 5, Math.PI / 5), randFloat(0.25, 0.75), true)
        }
      })
      controls.smoothTime = .5
      controls.moveTo(200, 200, 200, true)

      return scene
    }
  },
  mounted() {
    this.setupScene()
    console.log("loaded")
  }
}
</script>
