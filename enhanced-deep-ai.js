/**
 * Enhanced Deep AI for Snake Game
 * Combines rule-based intelligence with reinforcement learning
 */
class EnhancedDeepAI {
    constructor(gridWidth, gridHeight) {
        this.gridWidth = gridWidth;
        this.gridHeight = gridHeight;

        // Neural network parameters
        this.inputSize = 16;  // Enhanced state representation
        this.hiddenSize = 24; // Larger hidden layer for more complex patterns
        this.outputSize = 4;  // Four possible directions

        // Initialize neural network weights
        this.initializeWeights();

        // Learning parameters
        this.learningRate = 0.01;
        this.explorationRate = 0.1;
        this.discountFactor = 0.95;

        // Experience replay memory for training
        this.replayMemory = [];
        this.maxMemorySize = 10000;
        this.batchSize = 64;

        // Prioritized experience replay
        this.priorities = [];
        this.priorityAlpha = 0.6;  // How much prioritization to use (0 = none, 1 = full)
        this.priorityBeta = 0.4;   // Importance sampling correction (starts low, anneals to 1)

        // Training statistics
        this.trainingCount = 0;
        this.totalLoss = 0;
        this.averageLoss = 0;
        this.gamesPlayed = 0;
        this.highScore = 0;

        // Last state and action for training
        this.lastState = null;
        this.lastAction = null;
        this.lastHiddenLayer = null;

        // Performance tracking
        this.performanceHistory = [];
        this.evaluationInterval = 10; // Evaluate every 10 training sessions

        // Load saved model if available
        this.loadModel();

        console.log('Initialized enhanced deep learning AI');
    }

    // Initialize neural network weights with Xavier initialization
    initializeWeights() {
        // Xavier initialization scale factors
        const inputScale = Math.sqrt(2 / this.inputSize);
        const hiddenScale = Math.sqrt(2 / this.hiddenSize);

        // Input to hidden layer weights
        this.inputToHidden = Array(this.inputSize).fill().map(() =>
            Array(this.hiddenSize).fill().map(() => (Math.random() * 2 - 1) * inputScale)
        );

        // Hidden to output layer weights
        this.hiddenToOutput = Array(this.hiddenSize).fill().map(() =>
            Array(this.outputSize).fill().map(() => (Math.random() * 2 - 1) * hiddenScale)
        );

        // Bias terms
        this.hiddenBias = Array(this.hiddenSize).fill().map(() => (Math.random() * 2 - 1) * 0.1);
        this.outputBias = Array(this.outputSize).fill().map(() => (Math.random() * 2 - 1) * 0.1);

        console.log('Initialized neural network weights with Xavier initialization');
    }

    // Enhanced state representation
    getEnhancedState(state) {
        // Basic state features
        const basicFeatures = [
            state.foodDirX,                // Direction to food X (-1, 0, 1)
            state.foodDirY,                // Direction to food Y (-1, 0, 1)
            state.dangerUp ? 1 : 0,        // Danger in up direction
            state.dangerRight ? 1 : 0,     // Danger in right direction
            state.dangerDown ? 1 : 0,      // Danger in down direction
            state.dangerLeft ? 1 : 0,      // Danger in left direction
            state.loopDetected ? 1 : 0     // Loop detection
        ];

        // Enhanced features
        const enhancedFeatures = [
            // Normalized distance to food
            Math.abs(state.headX - state.foodX) / this.gridWidth,
            Math.abs(state.headY - state.foodY) / this.gridHeight,

            // Normalized position on grid
            state.headX / this.gridWidth,
            state.headY / this.gridHeight,

            // Distance to walls
            state.headX / this.gridWidth,                    // Distance to left wall (normalized)
            (this.gridWidth - 1 - state.headX) / this.gridWidth,  // Distance to right wall
            state.headY / this.gridHeight,                   // Distance to top wall
            (this.gridHeight - 1 - state.headY) / this.gridHeight, // Distance to bottom wall

            // Current direction one-hot encoding
            state.currentDirection === 'up' ? 1 : 0,
            state.currentDirection === 'right' ? 1 : 0,
            state.currentDirection === 'down' ? 1 : 0,
            state.currentDirection === 'left' ? 1 : 0
        ];

        // Combine all features
        return [...basicFeatures, ...enhancedFeatures];
    }

    // Forward pass through the neural network
    forward(state, currentDirection) {
        // Add current direction to state
        state.currentDirection = currentDirection;

        // Get enhanced state representation
        const inputArray = this.getEnhancedState(state);

        // Ensure inputToHidden is properly initialized
        if (!this.inputToHidden || !this.inputToHidden[0]) {
            console.error('Neural network weights not properly initialized');
            this.initializeWeights(); // Re-initialize weights if needed
        }

        // Forward pass through the neural network
        // Hidden layer
        const hiddenLayer = Array(this.hiddenSize).fill(0);
        for (let i = 0; i < this.hiddenSize; i++) {
            // Sum weighted inputs
            for (let j = 0; j < Math.min(inputArray.length, this.inputToHidden.length); j++) {
                if (this.inputToHidden[j] && this.inputToHidden[j][i] !== undefined) {
                    hiddenLayer[i] += inputArray[j] * this.inputToHidden[j][i];
                }
            }
            // Add bias
            hiddenLayer[i] += this.hiddenBias[i];
            // Apply ReLU activation function
            hiddenLayer[i] = Math.max(0, hiddenLayer[i]);
        }

        // Ensure hiddenToOutput is properly initialized
        if (!this.hiddenToOutput || !this.hiddenToOutput[0]) {
            console.error('Neural network weights not properly initialized');
            this.initializeWeights(); // Re-initialize weights if needed
        }

        // Output layer
        const outputLayer = Array(this.outputSize).fill(0);
        for (let i = 0; i < this.outputSize; i++) {
            // Sum weighted inputs from hidden layer
            for (let j = 0; j < hiddenLayer.length; j++) {
                if (this.hiddenToOutput[j] && this.hiddenToOutput[j][i] !== undefined) {
                    outputLayer[i] += hiddenLayer[j] * this.hiddenToOutput[j][i];
                }
            }
            // Add bias
            if (this.outputBias && this.outputBias[i] !== undefined) {
                outputLayer[i] += this.outputBias[i];
            }
        }

        // Apply softmax to get probabilities with numerical stability
        // First find the maximum value to prevent overflow
        const maxOutput = Math.max(...outputLayer);
        // Subtract max from each value and exponentiate
        const expValues = outputLayer.map(x => Math.exp(x - maxOutput));
        // Sum the exponentials
        const expSum = expValues.reduce((a, b) => a + b, 0) || 1; // Avoid division by zero
        // Normalize to get probabilities
        const actionProbabilities = expValues.map(x => x / expSum);

        // Check for NaN values and replace with uniform distribution if needed
        if (actionProbabilities.some(isNaN)) {
            console.warn('NaN detected in action probabilities, using uniform distribution');
            return {
                hiddenLayer,
                actionProbabilities: Array(this.outputSize).fill(1/this.outputSize)
            };
        }

        return {
            hiddenLayer,
            actionProbabilities
        };
    }

    // A* pathfinding algorithm to find path to food
    findPathToFood(state, checkCollisionFn) {
        const startPos = { x: state.headX, y: state.headY };
        const goalPos = { x: state.foodX, y: state.foodY };

        // If food is at the same position as head, no path needed
        if (startPos.x === goalPos.x && startPos.y === goalPos.y) {
            return [];
        }

        // Priority queue for A* algorithm
        const openSet = [];
        openSet.push({
            pos: startPos,
            g: 0, // Cost from start to current node
            h: this.manhattanDistance(startPos, goalPos), // Heuristic (estimated cost to goal)
            f: this.manhattanDistance(startPos, goalPos), // f = g + h
            parent: null
        });

        // Closed set to avoid revisiting nodes
        const closedSet = new Set();

        // Directions to explore
        const directions = [
            { x: 0, y: -1, name: 'up' },    // Up
            { x: 1, y: 0, name: 'right' },  // Right
            { x: 0, y: 1, name: 'down' },   // Down
            { x: -1, y: 0, name: 'left' }   // Left
        ];

        // A* search
        while (openSet.length > 0) {
            // Find node with lowest f score
            let lowestIndex = 0;
            for (let i = 1; i < openSet.length; i++) {
                if (openSet[i].f < openSet[lowestIndex].f) {
                    lowestIndex = i;
                }
            }

            const current = openSet[lowestIndex];

            // If we reached the goal, reconstruct and return the path
            if (current.pos.x === goalPos.x && current.pos.y === goalPos.y) {
                const path = [];
                let temp = current;
                while (temp.parent) {
                    path.push(temp.direction);
                    temp = temp.parent;
                }
                return path.reverse(); // Reverse to get path from start to goal
            }

            // Remove current from openSet and add to closedSet
            openSet.splice(lowestIndex, 1);
            closedSet.add(`${current.pos.x},${current.pos.y}`);

            // Check all neighbor directions
            for (const dir of directions) {
                const neighbor = {
                    x: current.pos.x + dir.x,
                    y: current.pos.y + dir.y
                };

                // Skip if neighbor is in closedSet or is a collision
                const neighborKey = `${neighbor.x},${neighbor.y}`;
                if (closedSet.has(neighborKey) || checkCollisionFn(neighbor)) {
                    continue;
                }

                // Calculate g score for this neighbor
                const gScore = current.g + 1;

                // Check if neighbor is already in openSet
                let neighborInOpenSet = false;
                for (let i = 0; i < openSet.length; i++) {
                    if (openSet[i].pos.x === neighbor.x && openSet[i].pos.y === neighbor.y) {
                        neighborInOpenSet = true;

                        // If this path is better, update the neighbor
                        if (gScore < openSet[i].g) {
                            openSet[i].g = gScore;
                            openSet[i].f = gScore + openSet[i].h;
                            openSet[i].parent = current;
                            openSet[i].direction = dir.name;
                        }
                        break;
                    }
                }

                // If neighbor is not in openSet, add it
                if (!neighborInOpenSet) {
                    const h = this.manhattanDistance(neighbor, goalPos);
                    openSet.push({
                        pos: neighbor,
                        g: gScore,
                        h: h,
                        f: gScore + h,
                        parent: current,
                        direction: dir.name
                    });
                }
            }
        }

        // No path found
        return null;
    }

    // Manhattan distance heuristic for A*
    manhattanDistance(pos1, pos2) {
        return Math.abs(pos1.x - pos2.x) + Math.abs(pos1.y - pos2.y);
    }

    // Check if a move would create a trap (enclosed area)
    wouldCreateTrap(state, direction, checkCollisionFn) {
        // Simulate the move
        const newPos = {
            x: state.headX + (direction === 'right' ? 1 : direction === 'left' ? -1 : 0),
            y: state.headY + (direction === 'down' ? 1 : direction === 'up' ? -1 : 0)
        };

        // If the move would cause a collision, it's not valid
        if (checkCollisionFn(newPos)) {
            return true;
        }

        // Check if this move creates an enclosed area
        // This is a simplified flood fill algorithm to check reachable cells
        const visited = new Set();
        const queue = [newPos];

        while (queue.length > 0) {
            const pos = queue.shift();
            const key = `${pos.x},${pos.y}`;

            if (visited.has(key)) continue;
            visited.add(key);

            // Check all four directions
            const neighbors = [
                { x: pos.x, y: pos.y - 1 }, // Up
                { x: pos.x + 1, y: pos.y }, // Right
                { x: pos.x, y: pos.y + 1 }, // Down
                { x: pos.x - 1, y: pos.y }  // Left
            ];

            for (const neighbor of neighbors) {
                // Skip if out of bounds or collision
                if (neighbor.x < 0 || neighbor.x >= this.gridWidth ||
                    neighbor.y < 0 || neighbor.y >= this.gridHeight ||
                    checkCollisionFn(neighbor)) {
                    continue;
                }

                queue.push(neighbor);
            }
        }

        // Calculate the percentage of grid that's reachable
        const totalCells = this.gridWidth * this.gridHeight;
        const reachableCells = visited.size;
        const reachablePercentage = reachableCells / totalCells;

        // If less than 50% of the grid is reachable, consider it a trap
        return reachablePercentage < 0.5;
    }

    // Get the best action based on the current state
    getAction(state, currentDirection, checkCollisionFn) {
        // Store current state for later training
        this.lastState = { ...state };

        // Try to find a path to food using A*
        const pathToFood = this.findPathToFood(state, checkCollisionFn);

        // If a path to food exists and it's not empty, follow it
        if (pathToFood && pathToFood.length > 0) {
            const nextDirection = pathToFood[0];

            // Check if following the path would create a trap
            if (!this.wouldCreateTrap(state, nextDirection, checkCollisionFn)) {
                this.lastAction = nextDirection;
                return nextDirection;
            }
        }

        // If no path to food or it would create a trap, use neural network
        const { hiddenLayer, actionProbabilities } = this.forward(state, currentDirection);
        this.lastHiddenLayer = [...hiddenLayer];

        // Get valid actions (no immediate collisions)
        const actions = ['up', 'right', 'down', 'left'];
        const validActions = actions.filter(action => {
            const newPos = {
                x: state.headX + (action === 'right' ? 1 : action === 'left' ? -1 : 0),
                y: state.headY + (action === 'down' ? 1 : action === 'up' ? -1 : 0)
            };
            return !checkCollisionFn(newPos);
        });

        // If no valid actions, choose the least bad one
        if (validActions.length === 0) {
            let bestAction = actions[0];
            let bestValue = actionProbabilities[0];

            for (let i = 1; i < actions.length; i++) {
                if (actionProbabilities[i] > bestValue) {
                    bestValue = actionProbabilities[i];
                    bestAction = actions[i];
                }
            }

            this.lastAction = bestAction;
            return bestAction;
        }

        // Apply epsilon-greedy exploration
        if (Math.random() < this.explorationRate) {
            // Explore: choose a random valid action
            const randomIndex = Math.floor(Math.random() * validActions.length);
            this.lastAction = validActions[randomIndex];
            return validActions[randomIndex];
        } else {
            // Exploit: choose the best valid action according to the model
            let bestValidAction = validActions[0];
            let bestValidProb = actionProbabilities[actions.indexOf(validActions[0])];

            for (let i = 1; i < validActions.length; i++) {
                const actionIndex = actions.indexOf(validActions[i]);
                if (actionProbabilities[actionIndex] > bestValidProb) {
                    bestValidProb = actionProbabilities[actionIndex];
                    bestValidAction = validActions[i];
                }
            }

            this.lastAction = bestValidAction;
            return bestValidAction;
        }
    }

    // Store experience with priority
    storeExperience(state, action, reward, nextState, done) {
        // Calculate priority based on reward magnitude
        const priority = Math.abs(reward) + 0.01; // Small constant to ensure non-zero priority

        // Store experience in replay memory
        this.replayMemory.push({
            state: { ...state }, // Clone state to avoid reference issues
            action,
            reward,
            nextState: nextState ? { ...nextState } : null,
            done
        });

        // Store priority
        this.priorities.push(priority);

        // Limit memory size
        if (this.replayMemory.length > this.maxMemorySize) {
            this.replayMemory.shift(); // Remove oldest experience
            this.priorities.shift();   // Remove oldest priority
        }
    }

    // Handle food eaten event - give high reward
    handleFoodEaten(state) {
        if (this.lastState && this.lastAction) {
            // Give a high reward for eating food
            this.storeExperience(
                this.lastState,
                this.lastAction,
                10.0, // High reward for eating food
                state,
                false // Not done yet
            );

            console.log('Food eaten! Stored positive experience with reward: 10.0');

            // Train immediately after eating food
            if (this.replayMemory.length >= this.batchSize) {
                this.trainOnBatch();
            }

            // Save model periodically after eating food
            if (Math.random() < 0.2) { // 20% chance to save after eating
                this.saveModel();
            }
        }
    }

    // Handle game over event - give negative reward
    handleGameOver(state, score) {
        if (this.lastState && this.lastAction) {
            // Update high score
            if (score > this.highScore) {
                this.highScore = score;
            }

            // Increment games played
            this.gamesPlayed++;

            // Calculate reward based on score
            // Higher scores get less negative rewards
            const normalizedScore = Math.min(score / 50, 1); // Normalize score (max 50 for full normalization)
            const reward = -10.0 + normalizedScore * 8.0; // Range from -10 to -2 based on score

            // Give a negative reward for game over
            this.storeExperience(
                this.lastState,
                this.lastAction,
                reward, // Negative reward for dying
                state,
                true // Game is done
            );

            console.log(`Game over! Score: ${score}, Stored negative experience with reward: ${reward.toFixed(2)}`);

            // Train immediately after game over
            if (this.replayMemory.length >= this.batchSize) {
                this.trainOnBatch();
            }

            // Always save model after game over
            this.saveModel();

            // Track performance
            this.performanceHistory.push({
                gamesPlayed: this.gamesPlayed,
                score: score,
                averageLoss: this.averageLoss,
                explorationRate: this.explorationRate
            });

            // Adjust exploration rate based on performance
            this.adjustExplorationRate();
        }
    }

    // Adjust exploration rate based on performance
    adjustExplorationRate() {
        // Decrease exploration rate over time, but keep a minimum
        this.explorationRate = Math.max(0.05, this.explorationRate * 0.995);

        // If we have enough performance history, check if we're improving
        if (this.performanceHistory.length >= 10) {
            const recentScores = this.performanceHistory.slice(-10).map(p => p.score);
            const avgRecentScore = recentScores.reduce((a, b) => a + b, 0) / recentScores.length;

            // If average score is low, increase exploration to try new strategies
            if (avgRecentScore < 5) {
                this.explorationRate = Math.min(0.3, this.explorationRate * 1.1);
            }
        }
    }

    // Train on a batch of experiences from replay memory with prioritization
    trainOnBatch() {
        // Skip if not enough samples
        if (this.replayMemory.length < this.batchSize) {
            return;
        }

        // Calculate sum of priorities
        const prioritySum = this.priorities.reduce((a, b) => a + b, 0);

        // Sample a batch of experiences based on priorities
        const batch = [];
        const indices = [];
        const weights = []; // Importance sampling weights

        for (let i = 0; i < this.batchSize; i++) {
            // Sample based on priority
            let value = Math.random() * prioritySum;
            let sum = 0;
            let index = 0;

            while (sum < value && index < this.priorities.length) {
                sum += Math.pow(this.priorities[index], this.priorityAlpha);
                index++;
            }

            index = Math.max(0, index - 1);
            indices.push(index);
            batch.push(this.replayMemory[index]);

            // Calculate importance sampling weight
            const probability = Math.pow(this.priorities[index], this.priorityAlpha) / prioritySum;
            weights.push(Math.pow(this.replayMemory.length * probability, -this.priorityBeta));
        }

        // Normalize weights
        const maxWeight = Math.max(...weights);
        const normalizedWeights = weights.map(w => w / maxWeight);

        let totalLoss = 0;

        // Train on each experience in the batch
        for (let i = 0; i < batch.length; i++) {
            const experience = batch[i];
            const { state, action, reward, nextState, done } = experience;
            const weight = normalizedWeights[i];

            // Skip if state or nextState is null
            if (!state || !nextState) continue;

            // Forward pass for current state
            const currentDirection =
                state.currentDirection ||
                (state.directionUp ? 'up' :
                 state.directionRight ? 'right' :
                 state.directionDown ? 'down' : 'left');

            const { actionProbabilities } = this.forward(state, currentDirection);

            // Forward pass for next state to get target values
            const nextDirection =
                nextState.currentDirection ||
                (nextState.directionUp ? 'up' :
                 nextState.directionRight ? 'right' :
                 nextState.directionDown ? 'down' : 'left');

            const { actionProbabilities: nextActionProbs } = this.forward(nextState, nextDirection);

            // Calculate target Q-value using Q-learning update rule
            const actionIndex = ['up', 'right', 'down', 'left'].indexOf(action);
            const maxNextQ = Math.max(...nextActionProbs);
            const targetQ = done ? reward : reward + this.discountFactor * maxNextQ;

            // Calculate loss (squared error)
            const currentQ = actionProbabilities[actionIndex];
            const error = targetQ - currentQ;
            const loss = error * error * weight; // Apply importance sampling weight
            totalLoss += loss;

            // Update priorities based on error
            this.priorities[indices[i]] = Math.abs(error) + 0.01;

            // Update weights for the action taken
            const { hiddenLayer } = this.forward(state, currentDirection);

            // Update output layer weights
            for (let j = 0; j < hiddenLayer.length; j++) {
                this.hiddenToOutput[j][actionIndex] +=
                    this.learningRate * error * hiddenLayer[j] * weight;
            }

            // Update output bias
            this.outputBias[actionIndex] += this.learningRate * error * weight;
        }

        // Update training statistics
        this.trainingCount++;
        this.totalLoss += totalLoss / this.batchSize;
        this.averageLoss = this.totalLoss / this.trainingCount;

        // Gradually increase beta for importance sampling
        this.priorityBeta = Math.min(1.0, this.priorityBeta + 0.001);

        // Log training progress occasionally
        if (this.trainingCount % 10 === 0) {
            console.log(`Training batch #${this.trainingCount}, Avg Loss: ${this.averageLoss.toFixed(4)}, Exploration: ${this.explorationRate.toFixed(4)}, Memory: ${this.replayMemory.length} samples`);
        }
    }

    // Get training statistics
    getTrainingStats() {
        return {
            memorySize: this.replayMemory.length,
            batchSize: this.batchSize,
            trainingCount: this.trainingCount,
            averageLoss: this.averageLoss,
            explorationRate: this.explorationRate,
            gamesPlayed: this.gamesPlayed,
            highScore: this.highScore
        };
    }

    // Save the model weights to localStorage
    saveModel() {
        try {
            const modelData = {
                inputToHidden: this.inputToHidden,
                hiddenToOutput: this.hiddenToOutput,
                hiddenBias: this.hiddenBias,
                outputBias: this.outputBias,
                trainingCount: this.trainingCount,
                totalLoss: this.totalLoss,
                averageLoss: this.averageLoss,
                explorationRate: this.explorationRate,
                replayMemorySize: this.replayMemory.length,
                gamesPlayed: this.gamesPlayed,
                highScore: this.highScore,
                performanceHistory: this.performanceHistory.slice(-20) // Save only last 20 entries
            };

            localStorage.setItem('enhanced-deep-ai-model', JSON.stringify(modelData));
            console.log(`Saved enhanced deep AI model to localStorage (Training: ${this.trainingCount}, Games: ${this.gamesPlayed}, High Score: ${this.highScore})`);
            return true;
        } catch (error) {
            console.error('Failed to save enhanced deep AI model:', error);
            return false;
        }
    }

    // Load the model weights from localStorage
    loadModel() {
        try {
            const modelData = JSON.parse(localStorage.getItem('enhanced-deep-ai-model'));
            if (modelData) {
                // Validate model structure before loading
                if (!modelData.inputToHidden || !modelData.hiddenToOutput) {
                    console.warn('Invalid model structure in localStorage, initializing new model');
                    this.initializeWeights();
                    return false;
                }

                // Check if dimensions match
                if (modelData.inputToHidden.length !== this.inputSize ||
                    modelData.hiddenToOutput.length !== this.hiddenSize) {
                    console.warn('Model dimensions mismatch, initializing new model');
                    this.initializeWeights();
                    return false;
                }

                // Load weights
                this.inputToHidden = modelData.inputToHidden;
                this.hiddenToOutput = modelData.hiddenToOutput;
                this.hiddenBias = modelData.hiddenBias || Array(this.hiddenSize).fill(0);
                this.outputBias = modelData.outputBias || Array(this.outputSize).fill(0);

                // Load training statistics if available
                if (modelData.trainingCount !== undefined) {
                    this.trainingCount = modelData.trainingCount;
                }
                if (modelData.totalLoss !== undefined) {
                    this.totalLoss = modelData.totalLoss;
                }
                if (modelData.averageLoss !== undefined) {
                    this.averageLoss = modelData.averageLoss;
                }
                if (modelData.explorationRate !== undefined) {
                    this.explorationRate = modelData.explorationRate;
                }
                if (modelData.gamesPlayed !== undefined) {
                    this.gamesPlayed = modelData.gamesPlayed;
                }
                if (modelData.highScore !== undefined) {
                    this.highScore = modelData.highScore;
                }
                if (modelData.performanceHistory !== undefined) {
                    this.performanceHistory = modelData.performanceHistory;
                }

                console.log(`Loaded enhanced deep AI model from localStorage (Training: ${this.trainingCount}, Games: ${this.gamesPlayed}, High Score: ${this.highScore})`);
                return true;
            }

            console.log('No saved model found, initializing new model');
            return false;
        } catch (error) {
            console.error('Failed to load enhanced deep AI model:', error);
            // Initialize new weights if loading fails
            this.initializeWeights();
            return false;
        }
    }
}
