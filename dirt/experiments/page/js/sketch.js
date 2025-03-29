let gridSize = 10;
let cols = 32;
let rows = 32;
let dataset = [];  // Array of grid states
let currentIndex = 0;
let playing = false;
let interval;
let isDragging;

let chart;
let chartData = {
    labels: Array.from({ length: 32 * 32 }, (_, i) => i), // Label for each grid cell
    datasets: [
        {
            label: "Grid Data (Red Channel)", 
            data: Array(32 * 32).fill(0), // Initial empty data
            borderColor: "red",
            backgroundColor: "rgba(255, 0, 0, 0.2)",
            fill: true
        }
    ]
};

function setup() {
    let c = createCanvas(cols * gridSize, rows * gridSize);
    c.parent("canvas-container");
    noStroke();
    
    generateDataset(); // Create example grid dataset
    
    // Add button event listeners
    document.getElementById('playPause').addEventListener('click', togglePlayPause);
    document.getElementById('next').addEventListener('click', nextGrid);
    document.getElementById('prev').addEventListener('click', prevGrid);

    setupChart();

    // Add slider event listener
    let timelineContainer = document.getElementById('timeline-container');
    
    timelineContainer.addEventListener('mousedown', (event) => {
        isDragging = true;
        updateTimeline(event);
    });

    window.addEventListener('mousemove', (event) => {
        if (isDragging) updateTimeline(event);
    });

    window.addEventListener('mouseup', () => {
        isDragging = false;
    });

    updateTimelineHandle();
}

function draw() {
    background(0);
    if (dataset.length > 0) {
        drawGrid(dataset[currentIndex]);
    }
    updateTimelineHandle();
    updateChart();
}

function setupChart() {
    let ctx = document.getElementById("lineChart").getContext("2d");
    chart = new Chart(ctx, {
        type: "line",
        data: chartData,
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: "Grid Cell Index" } },
                y: { title: { display: true, text: "Red Channel Value" } }
            }
        }
    });
}


function updateChart() {
    if (!chart) return;  // Prevent errors if the chart isn't ready

    let grid = dataset[currentIndex]; // Get the current grid state
    let redValues = grid.flat().map(c => red(c)); // Extract red channel values

    chartData.datasets[0].data = redValues; // Update chart data
    chart.update(); // Refresh chart
}



// Generate example dataset (replace with actual data loading)
function generateDataset() {
    for (let i = 0; i < 1000; i++) {  // 20 different grid states
        let grid = [];
        for (let r = 0; r < rows; r++) {
            let row = [];
            for (let c = 0; c < cols; c++) {
                row.push(color(random(255), random(255), random(255))); // Random colors
            }
            grid.push(row);
        }
        dataset.push(grid);
    }
}

function updateTimeline(event) {
  let rect = document.getElementById('timeline-container').getBoundingClientRect();
  let dragX = event.clientX - rect.left;
  let newIndex = Math.round((dragX / rect.width) * (dataset.length - 1));
  currentIndex = Math.max(0, Math.min(dataset.length - 1, newIndex));
  updateTimelineHandle();
}

function updateTimelineHandle() {
  let handle = document.getElementById('timeline-handle');
  let progress = (currentIndex / (dataset.length - 1)) * 100;
  handle.style.left = `${progress}%`;
}

// Draw a specific grid
function drawGrid(grid) {
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            fill(grid[r][c]);
            rect(c * gridSize, r * gridSize, gridSize, gridSize);
        }
    }
}

// Play/Pause toggle
function togglePlayPause() {
    playing = !playing;
    document.getElementById('playPause').textContent = playing ? "⏸️ Pause" : "▶️ Play";

    if (playing) {
        interval = setInterval(() => {
            nextGrid();
        }, 100); // Change every second
    } else {
        clearInterval(interval);
    }
}

// Move to next grid
function nextGrid() {
    currentIndex = (currentIndex + 1) % dataset.length;
}

// Move to previous grid
function prevGrid() {
    currentIndex = (currentIndex - 1 + dataset.length) % dataset.length;
}