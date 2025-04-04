// Deep learning AI decision making with simplified implementation (no TensorFlow dependency)
async function makeDeepAIDecision() {
    try {
        // Initialize the simplified deep AI if not already done
        if (!window.simplifiedDeepAI) {
            window.simplifiedDeepAI = new SimplifiedDeepAI(gridWidth, gridHeight);
            // Try to load saved model
            window.simplifiedDeepAI.loadModel();
        }
        
        // Get current state
        const head = snake[0];
        const state = getState(head);
        
        // Add head and food positions to state for learning
        state.headX = head.x;
        state.headY = head.y;
        state.foodX = food.x;
        state.foodY = food.y;
        
        // Get action from simplified deep AI
        const chosenAction = window.simplifiedDeepAI.getAction(state, direction, checkCollision);
        
        // Set the direction based on the chosen action
        switch (chosenAction) {
            case 'up':
                if (direction !== 'down') { // Prevent 180-degree turns
                    direction = 'up';
                } else {
                    // If trying to go in the opposite direction, use basic AI
                    makeBasicAIDecision();
                }
                break;
            case 'right':
                if (direction !== 'left') {
                    direction = 'right';
                } else {
                    makeBasicAIDecision();
                }
                break;
            case 'down':
                if (direction !== 'up') {
                    direction = 'down';
                } else {
                    makeBasicAIDecision();
                }
                break;
            case 'left':
                if (direction !== 'right') {
                    direction = 'left';
                } else {
                    makeBasicAIDecision();
                }
                break;
        }
        
        // Clear path visualization
        currentPath = [];
        
        // Periodically save the model
        if (Math.random() < 0.01) { // 1% chance to save
            window.simplifiedDeepAI.saveModel();
        }
    } catch (error) {
        console.error('Deep AI decision error:', error);
        // Show user-friendly message
        showTooltip('Deep AI error - using basic AI instead', 3000);
        // Fall back to basic AI if deep learning fails
        makeBasicAIDecision();
    }
}
