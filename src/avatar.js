// Avatar integration with ASL recognition API
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

class ASLAvatar {
  constructor(containerId, apiBaseUrl = 'http://localhost:5000') {
    // Configuration
    this.apiBaseUrl = apiBaseUrl;
    this.containerId = containerId;
    this.container = document.getElementById(containerId);
    
    if (!this.container) {
      console.error(`Container with ID "${containerId}" not found.`);
      return;
    }
    
    // State
    this.isInitialized = false;
    this.isLoading = false;
    this.currentAnimation = null;
    this.animationMixer = null;
    this.avatarModel = null;
    this.animationClips = {};
    this.clock = new THREE.Clock();
    
    // Initialize Three.js scene
    this.initScene();
  }
  
  // Initialize Three.js scene
  initScene() {
    // Get container dimensions
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf0f0f0);
    
    // Create camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    this.camera.position.set(0, 1.6, 2);
    this.camera.lookAt(0, 1, 0);
    
    // Create renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.container.appendChild(this.renderer.domElement);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 2, 3);
    directionalLight.castShadow = true;
    this.scene.add(directionalLight);
    
    // Add ground
    const groundGeometry = new THREE.PlaneGeometry(10, 10);
    const groundMaterial = new THREE.MeshStandardMaterial({ 
      color: 0xcccccc,
      roughness: 0.8,
      metalness: 0.2
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);
    
    // Add orbit controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 1, 0);
    this.controls.update();
    
    // Handle window resize
    window.addEventListener('resize', () => this.onWindowResize());
    
    // Start animation loop
    this.animate();
    
    this.isInitialized = true;
  }
  
  // Load avatar model
  async loadAvatarModel(modelPath) {
    if (!this.isInitialized) {
      console.error('Scene not initialized yet.');
      return;
    }
    
    this.isLoading = true;
    
    try {
      const loader = new GLTFLoader();
      
      // Load the model
      const gltf = await new Promise((resolve, reject) => {
        loader.load(
          modelPath,
          (gltf) => resolve(gltf),
          (xhr) => console.log(`Loading model: ${Math.floor(xhr.loaded / xhr.total * 100)}%`),
          (error) => reject(error)
        );
      });
      
      // Set up the model
      this.avatarModel = gltf.scene;
      this.avatarModel.scale.set(1, 1, 1);
      this.avatarModel.position.set(0, 0, 0);
      this.avatarModel.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });
      
      this.scene.add(this.avatarModel);
      
      // Set up animation mixer
      this.animationMixer = new THREE.AnimationMixer(this.avatarModel);
      
      // Store default animations from the model
      gltf.animations.forEach((clip) => {
        this.animationClips[clip.name] = clip;
      });
      
      console.log('Avatar model loaded successfully.');
      console.log('Available animations:', Object.keys(this.animationClips));
      
      this.isLoading = false;
      return true;
    } catch (error) {
      console.error('Error loading avatar model:', error);
      this.isLoading = false;
      return false;
    }
  }
  
  // Load specific animation for a sign
  async loadSignAnimation(signName, animationPath) {
    if (!this.avatarModel) {
      console.error('Avatar model not loaded yet.');
      return false;
    }
    
    try {
      const loader = new GLTFLoader();
      
      // Load the animation
      const gltf = await new Promise((resolve, reject) => {
        loader.load(
          animationPath,
          (gltf) => resolve(gltf),
          (xhr) => console.log(`Loading animation: ${Math.floor(xhr.loaded / xhr.total * 100)}%`),
          (error) => reject(error)
        );
      });
      
      // Store the animation
      if (gltf.animations && gltf.animations.length > 0) {
        this.animationClips[signName] = gltf.animations[0];
        console.log(`Animation for sign "${signName}" loaded successfully.`);
        return true;
      } else {
        console.error(`No animations found in ${animationPath}`);
        return false;
      }
    } catch (error) {
      console.error(`Error loading animation for sign "${signName}":`, error);
      return false;
    }
  }
  
  // Play a sign animation
  playSignAnimation(signName, loop = false) {
    if (!this.animationMixer || !this.animationClips[signName]) {
      console.error(`Animation for sign "${signName}" not found.`);
      return false;
    }
    
    // Stop current animation if any
    if (this.currentAnimation) {
      this.currentAnimation.stop();
    }
    
    // Play new animation
    this.currentAnimation = this.animationMixer.clipAction(this.animationClips[signName]);
    this.currentAnimation.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce);
    this.currentAnimation.clampWhenFinished = !loop;
    this.currentAnimation.reset().play();
    
    console.log(`Playing animation for sign "${signName}"`);
    return true;
  }
  
  // Send feature sequence to the API for prediction
  async predictSign(featureSequence) {
    try {
      const response = await fetch(`${this.apiBaseUrl}/predict-sequence`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          features: featureSequence
        })
      });
      
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      
      const result = await response.json();
      
      // Play animation based on predicted sign
      if (result.class_name) {
        this.playSignAnimation(result.class_name);
        return result;
      } else {
        console.error('No class name in prediction result:', result);
        return null;
      }
    } catch (error) {
      console.error('Error predicting sign:', error);
      return null;
    }
  }
  
  // Upload a .pkl file to the API for prediction
  async uploadPklFile(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${this.apiBaseUrl}/predict-sequence`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      
      const result = await response.json();
      
      // Play animation based on predicted sign
      if (result.class_name) {
        this.playSignAnimation(result.class_name);
        return result;
      } else {
        console.error('No class name in prediction result:', result);
        return null;
      }
    } catch (error) {
      console.error('Error uploading .pkl file:', error);
      return null;
    }
  }
  
  // Convert English text to ASL gloss and play animations
  async convertTextToSigns(text) {
    try {
      // First, convert text to ASL gloss
      const glossResponse = await fetch(`${this.apiBaseUrl}/speech-to-gloss`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: text
        })
      });
      
      if (!glossResponse.ok) {
        throw new Error(`API request failed with status ${glossResponse.status}`);
      }
      
      const glossResult = await glossResponse.json();
      const aslGloss = glossResult.asl_gloss;
      
      console.log(`ASL Gloss: ${aslGloss}`);
      
      // Split gloss into individual signs
      const signs = aslGloss.split(' ');
      
      // Play each sign animation in sequence
      for (let i = 0; i < signs.length; i++) {
        const sign = signs[i].trim();
        if (sign) {
          await new Promise((resolve) => {
            if (this.animationClips[sign]) {
              this.playSignAnimation(sign);
              
              // Wait for animation to complete or timeout
              const checkAnimation = () => {
                if (!this.currentAnimation || !this.currentAnimation.isRunning()) {
                  resolve();
                } else {
                  setTimeout(checkAnimation, 100);
                }
              };
              
              setTimeout(checkAnimation, 1000); // Give animation time to start
            } else {
              console.warn(`No animation found for sign "${sign}"`);
              resolve();
            }
          });
        }
      }
      
      return aslGloss;
    } catch (error) {
      console.error('Error converting text to signs:', error);
      return null;
    }
  }
  
  // Handle window resize
  onWindowResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }
  
  // Animation loop
  animate() {
    requestAnimationFrame(() => this.animate());
    
    // Update controls
    this.controls.update();
    
    // Update animation mixer
    if (this.animationMixer) {
      this.animationMixer.update(this.clock.getDelta());
    }
    
    // Render scene
    this.renderer.render(this.scene, this.camera);
  }
}

// Export the class
export default ASLAvatar; 