import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'

/**
 * 3D FIFA World Cup Trophy on a rotating circular pedestal in a dark scene.
 * Features:
 * - Natural 3D model textures and metallic shading under spotlight
 * - Rotating circular metallic turntable surface with Lime Green neon tracks
 * - Responsive cursor flow animation with smooth inertia
 */
export default function TrophyScene() {
  const mountRef = useRef(null)

  useEffect(() => {
    const container = mountRef.current
    if (!container) return

    // 1. Scene & Camera Setup
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)

    const camera = new THREE.PerspectiveCamera(
      42,
      window.innerWidth / window.innerHeight,
      0.1,
      100
    )
    camera.position.set(0, 2.2, 8.8)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.3
    renderer.shadowMap.enabled = true
    container.appendChild(renderer.domElement)

    // 2. Trophy Group & GLTF Model Loading
    const trophyGroup = new THREE.Group()
    scene.add(trophyGroup)

    const textureLoader = new THREE.TextureLoader()
    const baseColorMap = textureLoader.load('/textures/world+cup+trophy_basecolor.jpg_0.jpeg')
    baseColorMap.colorSpace = THREE.SRGBColorSpace
    const normalMap = textureLoader.load('/textures/world+cup+trophy_normal.jpg_2.jpeg')

    const loader = new GLTFLoader()
    loader.load(
      '/models/trophy.glb',
      (gltf) => {
        const model = gltf.scene

        // Normalize size and center the model
        const box = new THREE.Box3().setFromObject(model)
        const size = box.getSize(new THREE.Vector3())
        const center = box.getCenter(new THREE.Vector3())

        const maxDim = Math.max(size.x, size.y, size.z)
        const scale = 4.8 / maxDim
        model.scale.set(scale, scale, scale)

        // Center vertically so base rests neatly on the circular pedestal
        model.position.x = -center.x * scale
        model.position.y = -box.min.y * scale - 1.15
        model.position.z = -center.z * scale

        model.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true
            child.receiveShadow = true
            if (child.material) {
              if (!child.material.map) {
                child.material.map = baseColorMap
              }
              if (!child.material.normalMap) {
                child.material.normalMap = normalMap
              }
              child.material.metalness = Math.max(child.material.metalness ?? 0, 0.85)
              child.material.roughness = Math.min(child.material.roughness ?? 0.35, 0.35)
              child.material.needsUpdate = true
            }
          }
        })

        trophyGroup.add(model)
      },
      undefined,
      (error) => {
        console.warn('Could not load /models/trophy.glb:', error)
      }
    )

    // 3. Circular Moving Surface (Rotating Pedestal Turntable)
    const turntableGroup = new THREE.Group()
    turntableGroup.position.set(0, -1.35, 0)
    turntableGroup.scale.set(1.2, 1.2, 1.2)

    const turntableMaterial = new THREE.MeshStandardMaterial({
      color: 0x121212,
      roughness: 0.5,
      metalness: 0.8,
    })

    const ringAccentMaterial = new THREE.MeshStandardMaterial({
      color: 0xb0e454,
      roughness: 0.3,
      metalness: 0.9,
      emissive: 0x163008,
      emissiveIntensity: 0.4,
    })

    // Main circular base disc
    const pedestal = new THREE.Mesh(
      new THREE.CylinderGeometry(3.6, 3.8, 0.22, 64),
      turntableMaterial
    )
    pedestal.receiveShadow = true
    turntableGroup.add(pedestal)

    // Outer accent ring
    const outerRing = new THREE.Mesh(
      new THREE.TorusGeometry(3.65, 0.025, 16, 80),
      ringAccentMaterial
    )
    outerRing.rotation.x = Math.PI / 2
    outerRing.position.y = 0.11
    turntableGroup.add(outerRing)

    // Middle concentric track
    const midRing = new THREE.Mesh(
      new THREE.TorusGeometry(2.4, 0.015, 12, 64),
      ringAccentMaterial
    )
    midRing.rotation.x = Math.PI / 2
    midRing.position.y = 0.11
    turntableGroup.add(midRing)

    // Inner base rim
    const innerRing = new THREE.Mesh(
      new THREE.TorusGeometry(1.48, 0.02, 12, 48),
      ringAccentMaterial
    )
    innerRing.rotation.x = Math.PI / 2
    innerRing.position.y = 0.11
    turntableGroup.add(innerRing)

    // Radial spokes on turntable surface to show movement clearly
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2
      const spoke = new THREE.Mesh(
        new THREE.BoxGeometry(0.02, 0.01, 1.8),
        ringAccentMaterial
      )
      spoke.position.set(
        Math.sin(angle) * 2.5,
        0.115,
        Math.cos(angle) * 2.5
      )
      spoke.rotation.y = angle
      turntableGroup.add(spoke)
    }

    scene.add(turntableGroup)

    // 4. Lighting Setup (Dramatic Spotlight & Subtle Fill in Darkness)
    const spotlight = new THREE.SpotLight(0xfffae0, 26, 28, Math.PI / 4.2, 0.45, 1.2)
    spotlight.position.set(3, 11, 5)
    spotlight.target = trophyGroup
    spotlight.castShadow = true
    scene.add(spotlight)

    const fillSpot = new THREE.SpotLight(0xe5f8d0, 10, 20, Math.PI / 4, 0.7, 1.2)
    fillSpot.position.set(-5, 8, -4)
    fillSpot.target = trophyGroup
    scene.add(fillSpot)

    const rimLight = new THREE.PointLight(0xb0e454, 6, 12)
    rimLight.position.set(-3.5, 1.5, 2.5)
    scene.add(rimLight)

    const ambientLight = new THREE.AmbientLight(0x383838, 1.6)
    scene.add(ambientLight)

    // 5. Cursor Animation with Flow & Inertia
    let targetRotationY = 0
    let targetRotationX = 0
    let currentRotationY = 0
    let currentRotationX = 0

    const onMouseMove = (e) => {
      const nx = (e.clientX / window.innerWidth) * 2 - 1
      const ny = (e.clientY / window.innerHeight) * 2 - 1

      targetRotationY = nx * 1.8
      targetRotationX = ny * 0.28
    }

    window.addEventListener('mousemove', onMouseMove, { passive: true })

    // 6. Render Loop
    let animationFrameId
    let lastTime = performance.now()

    const animate = (currentTime) => {
      animationFrameId = requestAnimationFrame(animate)
      const delta = (currentTime - lastTime) / 1000
      lastTime = currentTime

      // Circular surface turntable rotates continuously
      turntableGroup.rotation.y += (delta || 0.016) * 0.25

      // Trophy rotates towards cursor direction with smooth flow and settles
      currentRotationY += (targetRotationY - currentRotationY) * 0.045
      currentRotationX += (targetRotationX - currentRotationX) * 0.045

      trophyGroup.rotation.y = currentRotationY
      trophyGroup.rotation.x = currentRotationX

      // Spotlight subtly tracks for realistic glint
      spotlight.position.x = 3 + currentRotationY * 0.7

      renderer.render(scene, camera)
    }
    animate(performance.now())

    // 7. Resize Handler
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    }
    window.addEventListener('resize', onResize)

    // Cleanup
    return () => {
      cancelAnimationFrame(animationFrameId)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('resize', onResize)
      if (container && renderer.domElement) {
        container.removeChild(renderer.domElement)
      }
      renderer.dispose()
    }
  }, [])

  return (
    <div
      ref={mountRef}
      className="trophy-canvas-container"
      aria-hidden="true"
    />
  )
}
