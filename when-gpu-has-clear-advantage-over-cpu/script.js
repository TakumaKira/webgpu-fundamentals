const INPUT_SIZE = 100000;
const ITERATIONS_PER_INPUT = 1000;
const MAX_GPU_PARALLEL_EXECUTION = 256;

console.log('INPUT_SIZE', INPUT_SIZE);
console.log('ITERATIONS_PER_INPUT', ITERATIONS_PER_INPUT);
console.log('MAX_GPU_PARALLEL_EXECUTION', MAX_GPU_PARALLEL_EXECUTION);

async function main() {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice();
  if (!device) {
    fail('need a browser that supports WebGPU');
    return;
  }

  // Log device limits to see max workgroup size
  console.log('GPU Limits:', {
    maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: device.limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: device.limits.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
  });

  const safeMaximumWorkgroupSize = Math.min(MAX_GPU_PARALLEL_EXECUTION, device.limits.maxComputeWorkgroupSizeX);
  console.log('safeMaximumWorkgroupSize', safeMaximumWorkgroupSize);

  const module = device.createShaderModule({
    label: 'doubling compute module',
    code: /* wgsl */ `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(${safeMaximumWorkgroupSize})
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x;
        // Normalize the input to 0-1
        var result = input[i] / 100;

        // Intensive computation to show GPU advantage
        // Overriding the result with trigonometric functions many times shows clear GPU advantage
        for (var j = 0u; j < ${ITERATIONS_PER_INPUT}u; j++) {
          // Cut the result to 2 decimal places to reduce the precision loss over many iterations
          result = floor(result * 100) / 100;
          // Pass the result converted to radian range
          result = sin(result * 3.14);
          // Now the result is between -1 and 1
        }

        output[i] = result;
      }
    `,
  });

  function runCPUComputation(input) {
    const output = new Float32Array(input.length);
  
    for (let i = 0; i < input.length; i++) {
      // Normalize the input to 0-1
      let result = input[i] / 100;
  
      // Same intensive computation as GPU
      // Overriding the result with trigonometric functions many times shows clear GPU advantage
      for (let j = 0; j < ITERATIONS_PER_INPUT; j++) {
        // Cut the result to 2 decimal places to reduce the precision loss over many iterations
        result = Math.floor(result * 100) / 100;
        // Pass the result converted to radian range
        result = Math.sin(result * 3.14);
        // Now the result is between -1 and 1
      }
  
      output[i] = result;
    }
  
    return output;
  }
  
  const pipeline = device.createComputePipeline({
    label: 'doubling compute pipeline',
    layout: 'auto',
    compute: {
      module,
    },
  });

  const input = new Float32Array(INPUT_SIZE);
  for (let i = 0; i < input.length; i++) {
    input[i] = Math.random() * 100;
  }
  console.log('input', input);

  // create a buffer on the GPU to hold our computation input
  const inputBuffer = device.createBuffer({
    label: 'input buffer',
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // create a buffer on the GPU to hold our computation output
  const outputBuffer = device.createBuffer({
    label: 'output buffer',
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // create a buffer on the GPU to get a copy of the results
  const resultBuffer = device.createBuffer({
    label: 'result buffer',
    size: input.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });

  // Setup a bindGroup to tell the shader which
  // buffers to use for the computation
  const bindGroup = device.createBindGroup({
    label: 'bindGroup for input and output buffers',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  });

  // Start timing GPU computation (including data transfer)
  const gpuStartTime = performance.now();

  // Copy our input data to GPU buffer
  device.queue.writeBuffer(inputBuffer, 0, input);

  // Encode commands to do the computation
  const encoder = device.createCommandEncoder({
    label: 'doubling encoder',
  });
  const pass = encoder.beginComputePass({
    label: 'doubling compute pass',
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(input.length / MAX_GPU_PARALLEL_EXECUTION));
  pass.end();

  // Encode a command to copy the results to a mappable buffer.
  encoder.copyBufferToBuffer(outputBuffer, 0, resultBuffer, 0, resultBuffer.size);

  // Finish encoding and submit the commands
  const commandBuffer = encoder.finish();

  // Submit and wait for GPU to complete
  device.queue.submit([commandBuffer]);
  await device.queue.onSubmittedWorkDone();

  const gpuEndTime = performance.now();
  const gpuTime = gpuEndTime - gpuStartTime;

  // Read the results (outside timing since we excluded it from CPU too)
  await resultBuffer.mapAsync(GPUMapMode.READ);
  const gpuResult = new Float32Array(resultBuffer.getMappedRange()).slice();
  
  console.log('GPU result', gpuResult);

  // Clean up GPU buffers before running CPU computation
  resultBuffer.unmap();

  const cpuStartTime = performance.now();
  // Run CPU computation
  const cpuResult = runCPUComputation(input);
  const cpuEndTime = performance.now();
  const cpuTime = cpuEndTime - cpuStartTime;

  console.log('CPU result', cpuResult);

  // Display performance comparison
  displayResults(cpuTime, gpuTime);
}
main();

function displayResults(cpuTime, gpuTime) {
  const speedup = cpuTime / gpuTime;
  const resultsDiv = document.getElementById('results');

  resultsDiv.innerHTML = `
    <h2>Performance Comparison</h2>
    <p>CPU Time: ${cpuTime.toFixed(2)} ms</p>
    <p>GPU Time: ${gpuTime.toFixed(2)} ms</p>
    <p>Speedup: ${speedup.toFixed(2)}x (GPU is ${speedup.toFixed(2)} times faster)</p>
    <p>See console for more details</p>
  `;

  console.log(`\n=== Performance Comparison ===`);
  console.log(`CPU Time: ${cpuTime.toFixed(2)} ms`);
  console.log(`GPU Time: ${gpuTime.toFixed(2)} ms`);
  console.log(`Speedup: ${speedup.toFixed(2)}x`);
}