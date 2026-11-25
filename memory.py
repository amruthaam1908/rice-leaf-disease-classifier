import tensorflow as tf
import platform

print(f"System: {platform.system()} {platform.version()}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Python: {platform.python_version()}")

# Check GPU availability
print("\n" + "="*50)
print("GPU DETECTION RESULTS")
print("="*50)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ TensorFlow found {len(gpus)} GPU(s)")
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}:")
        print(f"  Name: {gpu.name}")
        print(f"  Type: {gpu.device_type}")
        
        # Try to get memory info
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                print(f"  Compute Capability: {details.get('compute_capability', 'Unknown')}")
        except:
            pass
        
        # Try to get memory info via TensorFlow 2.x method
        try:
            from tensorflow.python.client import device_lib
            local_devices = device_lib.list_local_devices()
            for device in local_devices:
                if 'GPU' in device.name:
                    print(f"  Memory: {device.memory_limit / 1e9:.2f} GB")
        except:
            pass
else:
    print("❌ NO GPU detected by TensorFlow")
    print("\nThis means:")
    print("1. You're using CPU only (integrated graphics)")
    print("2. Or NVIDIA drivers aren't installed properly")
    print("3. Or you have an AMD GPU (less compatible with TF)")
    
print("\n" + "="*50)
print("CPU/DEVICE INFO")
print("="*50)

# What device TensorFlow will use
print(f"Available devices: {tf.config.list_physical_devices()}")

# Test where operations run
print("\nTesting computation device...")
with tf.device('/CPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f"CPU test passed: {c.numpy()}")