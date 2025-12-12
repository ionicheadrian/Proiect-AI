// renderer.js - Renderer Process Script
let currentQuestion = null;
let currentAnswer = null;

// Elements
const categorySection = document.getElementById('category-section');
const questionSection = document.getElementById('question-section');
const resultSection = document.getElementById('result-section');

const categorySelect = document.getElementById('category');
const generateBtn = document.getElementById('generate-btn');
const loadingDiv = document.getElementById('loading');

const questionText = document.getElementById('question-text');
const userAnswerInput = document.getElementById('user-answer');
const submitBtn = document.getElementById('submit-btn');

const scoreDiv = document.getElementById('score');
const feedbackDiv = document.getElementById('feedback');
const correctAnswerText = document.getElementById('correct-answer-text');
const userAnswerText = document.getElementById('user-answer-text');
const tryAgainBtn = document.getElementById('try-again-btn');

// Generate Question
generateBtn.addEventListener('click', async () => {
    const category = categorySelect.value;
    
    if (!category) {
        alert('Please select a category!');
        return;
    }
    
    generateBtn.disabled = true;
    loadingDiv.style.display = 'block';
    
    try {
        const result = await window.api.generateQuestion(category);
        
        if (result.success) {
            currentQuestion = result.question;
            currentAnswer = result.answer;
            
            questionText.textContent = currentQuestion;
            
            // Switch to question section
            categorySection.classList.remove('active');
            questionSection.classList.add('active');
        } else {
            alert('Failed to generate question: ' + result.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        generateBtn.disabled = false;
        loadingDiv.style.display = 'none';
    }
});

// Submit Answer
submitBtn.addEventListener('click', () => {
    const userAnswer = userAnswerInput.value.trim();
    
    if (!userAnswer) {
        alert('Please enter your answer!');
        return;
    }
    
    const score = calculateSimilarity(userAnswer, currentAnswer);
    
    // Display score
    scoreDiv.textContent = score + '%';
    
    // Apply color class
    scoreDiv.className = 'score';
    if (score >= 80) {
        scoreDiv.classList.add('excellent');
        feedbackDiv.textContent = 'Excellent!';
    } else if (score >= 60) {
        scoreDiv.classList.add('good');
        feedbackDiv.textContent = 'Good effort!';
    } else {
        scoreDiv.classList.add('poor');
        feedbackDiv.textContent = 'Keep practicing!';
    }
    
    // Display answers
    correctAnswerText.textContent = currentAnswer;
    userAnswerText.textContent = userAnswer;
    
    // Switch to result section
    questionSection.classList.remove('active');
    resultSection.classList.add('active');
});

// Try Again
tryAgainBtn.addEventListener('click', () => {
    // Reset form
    categorySelect.value = '';
    userAnswerInput.value = '';
    currentQuestion = null;
    currentAnswer = null;
    
    // Switch back to category section
    resultSection.classList.remove('active');
    categorySection.classList.add('active');
});

// Calculate Similarity
function calculateSimilarity(userAnswer, correctAnswer) {
    function normalize(text) {
        return text.toLowerCase()
            .trim()
            .replace(/[^\w\s]/g, '');
    }
    
    const userNorm = normalize(userAnswer);
    const correctNorm = normalize(correctAnswer);
    
    // Exact match
    if (userNorm === correctNorm) {
        return 100;
    }
    
    // Sequence similarity
    const seqScore = sequenceSimilarity(userNorm, correctNorm) * 100;
    
    // Word overlap
    const userWords = new Set(userNorm.split(/\s+/));
    const correctWords = new Set(correctNorm.split(/\s+/));
    
    const intersection = new Set([...userWords].filter(x => correctWords.has(x)));
    const union = new Set([...userWords, ...correctWords]);
    
    const wordScore = union.size > 0 ? (intersection.size / union.size) * 100 : 0;
    
    // Weighted combination
    const finalScore = (seqScore * 0.6) + (wordScore * 0.4);
    
    return Math.round(finalScore);
}

function sequenceSimilarity(s1, s2) {
    const longer = s1.length > s2.length ? s1 : s2;
    const shorter = s1.length > s2.length ? s2 : s1;
    
    if (longer.length === 0) {
        return 1.0;
    }
    
    const editDistance = levenshteinDistance(longer, shorter);
    return (longer.length - editDistance) / longer.length;
}

function levenshteinDistance(s1, s2) {
    const costs = [];
    for (let i = 0; i <= s1.length; i++) {
        let lastValue = i;
        for (let j = 0; j <= s2.length; j++) {
            if (i === 0) {
                costs[j] = j;
            } else if (j > 0) {
                let newValue = costs[j - 1];
                if (s1.charAt(i - 1) !== s2.charAt(j - 1)) {
                    newValue = Math.min(
                        Math.min(newValue, lastValue),
                        costs[j]
                    ) + 1;
                }
                costs[j - 1] = lastValue;
                lastValue = newValue;
            }
        }
        if (i > 0) {
            costs[s2.length] = lastValue;
        }
    }
    return costs[s2.length];
}
