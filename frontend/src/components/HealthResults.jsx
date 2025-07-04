import React, { useState } from 'react';
import './HealthResults.css';
import LoginRegisterModal from './LoginRegisterModal';
import imageCompression from 'browser-image-compression';

const HealthResults = ({ analysisData, onSaveData, onBackToHome }) => {
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [selectedIngredient, setSelectedIngredient] = useState(null);
  const [isIngredientPopupOpen, setIsIngredientPopupOpen] = useState(false);
  const [ingredientData, setIngredientData] = useState(null);
  const [ingredientLoading, setIngredientLoading] = useState(false);
  const [ingredientError, setIngredientError] = useState(null);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [saveCompleted, setSaveCompleted] = useState(false);
  const isLoggedIn = localStorage.getItem('token') !== null;

  const API_BASE = 'https://donut-backend-o6ef.onrender.com';

  const handleAnalyzeAnother = async (file) => {
    if (file && !loading) {
      setLoading(true);
      
      const options = {
        maxSizeMB: 0.07,
        maxWidthOrHeight: 512,
        useWebWorker: true,
        fileType: 'image/jpeg',
      };

      try {
        const compressedFile = await imageCompression(file, options);
        const formData = new FormData();
        formData.append('file', compressedFile, file.name);

        const response = await fetch(`${API_BASE}/upload`, {
          method: 'POST',
          body: formData,
        });
        
        const data = await response.json();
        
        if (response.ok) {
          // Set currentView to 'donut-scene' in localStorage to trigger the animation
          localStorage.setItem('currentView', 'donut-scene');
          // Reload the page to show the donut animation with the new data
          window.location.reload();
        } else {
          setMessage(data.error || 'upload failed');
        }
      } catch (error) {
        console.error('Compression or Upload Error:', error);
        setMessage('an error occurred during compression or upload.');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleCameraClick = () => {
    document.getElementById('fileInput').click();
  };

  if (!analysisData) return null;

  const nutritionData = analysisData.nutrition_data || {};
  const ingredients = analysisData.ingredients || [];
  const foodName = analysisData.food_name || 'Unknown Food';

  const handleIngredientClick = (ingredient) => {
    if (!localStorage.getItem('token')) {
      setShowLoginModal(true);
      return;
    }
    setSelectedIngredient(ingredient);
    setIsIngredientPopupOpen(true);
    fetchIngredientData(ingredient);
  };

  const closeIngredientPopup = () => {
    setIsIngredientPopupOpen(false);
    setSelectedIngredient(null);
    setIngredientData(null);
    setIngredientError(null);
  };

  const fetchIngredientData = async (ingredient) => {
    setIngredientLoading(true);
    setIngredientError(null);
    
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        setIngredientError('Please log in to analyze ingredients');
        setIngredientLoading(false);
        return;
      }

      const response = await fetch(`${API_BASE}/analyze-ingredient`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ ingredient_name: ingredient }),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setIngredientData(data);
      } else {
        // Log the error details for debugging
        console.error('Ingredient analysis failed:', data);
        setIngredientError(data.detail || 'Failed to analyze ingredient');
      }
    } catch (error) {
      // Log the actual error for debugging
      console.error('Ingredient analysis error:', error);
      setIngredientError('Error analyzing ingredient');
    }
    
    setIngredientLoading(false);
  };

  const categories = [
    {
      id: 'nutrition',
      title: 'nutrition facts',
      icon: '🍎',
      data: nutritionData
    },
    {
      id: 'ingredients',
      title: 'ingredients',
      icon: '🧪',
      data: ingredients
    }
  ];

  const renderCategoryContent = (category) => {
    switch (category.id) {
      case 'nutrition':
        return (
          <div className="nutrition-grid">
            {Object.entries(category.data).map(([key, value]) => (
              <div key={key} className="nutrition-item">
                <span className="nutrition-label">{key.replace(/_/g, ' ').toUpperCase()}</span>
                <span className="nutrition-value">{value}</span>
              </div>
            ))}
          </div>
        );
      
      case 'ingredients':
        return (
          <div className="ingredients-list">
            {category.data.map((ingredient, index) => (
              <div 
                key={index} 
                className="ingredient-item"
                onClick={() => handleIngredientClick(ingredient)}
              >
                {ingredient}
              </div>
            ))}
          </div>
        );
      
      default:
        return null;
    }
  };

  const handleSave = async (consumed) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/save-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify({
          filename: 'food-analysis.jpg',
          analysis: analysisData,
          consumed: consumed
        }),
      });

      if (response.ok) {
        setMessage('Successfully saved to history!');
        setSaveCompleted(true);
      } else {
        console.error('Failed to save analysis');
        setMessage('Failed to save to history');
      }
    } catch (error) {
      console.error('Error saving analysis:', error);
      setMessage('Error saving to history');
    }
    setLoading(false);
  };

  return (
    <div className="health-results">
      {/* Hidden camera input */}
      <input
        type="file"
        id="fileInput"
        accept="image/*"
        capture="environment"
        onChange={(e) => {
          const file = e.target.files[0];
          if (file) {
            handleAnalyzeAnother(file);
          }
        }}
        style={{ display: 'none' }}
      />

      <div className="food-title">
        <h2>{foodName}</h2>
      </div>

      <div className="health-categories">
        {categories.map((category) => (
          <button
            key={category.id}
            className={`category-btn ${selectedCategory === category.id ? 'active' : ''}`}
            onClick={() => setSelectedCategory(selectedCategory === category.id ? null : category.id)}
          >
            <span className="category-icon">{category.icon}</span>
            <span className="category-title">{category.title}</span>
          </button>
        ))}
      </div>

      {selectedCategory && (
        <div className="category-details">
          <div className="details-content">
            {renderCategoryContent(categories.find(cat => cat.id === selectedCategory))}
          </div>
        </div>
      )}

      <div className="save-section">
        {isLoggedIn ? (
          <>
            <p className="save-prompt">what would you like to do with this analysis?</p>
            <div className="save-buttons">
              <button 
                className="save-btn eaten" 
                onClick={() => handleSave(true)} 
                disabled={loading || saveCompleted}
              >
                {loading ? 'saving...' : 'save to eaten'}
              </button>
              <button 
                className="save-btn avoided" 
                onClick={() => handleSave(false)} 
                disabled={loading || saveCompleted}
              >
                {loading ? 'saving...' : 'save to avoided'}
              </button>
            </div>
            {message && (
              <div className={`save-message ${message.includes('Successfully') ? 'success' : 'error'}`}>
                {message}
              </div>
            )}
          </>
        ) : (
          <div className="save-buttons">
            <button className="save-btn" onClick={() => setShowLoginModal(true)}>
              Login to Save History
            </button>
          </div>
        )}
      </div>

      <div className="results-actions">
        <button className="action-btn" onClick={handleCameraClick}>
          analyze another food
        </button>
      </div>

      {/* Ingredient Popup */}
      {isIngredientPopupOpen && (
        <div className="ingredient-popup-overlay" onClick={closeIngredientPopup}>
          <div className="ingredient-popup" onClick={e => e.stopPropagation()}>
            <button className="close-btn" onClick={closeIngredientPopup}>×</button>
            <h3>{selectedIngredient}</h3>
            
            {ingredientLoading && (
              <div className="loading-message">analyzing ingredient...</div>
            )}
            
            {ingredientError && (
              <div className="error-message">{ingredientError}</div>
            )}
            
            {ingredientData && (
              <div className="ingredient-details">
                <div className="detail-section">
                  <h4>pronunciation</h4>
                  <p>{ingredientData.pronunciation}</p>
                </div>
                
                <div className="detail-section">
                  <h4>purpose</h4>
                  <p>{ingredientData.purpose}</p>
                </div>

                <div className="detail-section">
                  <h4>commonly found in</h4>
                  <p>{ingredientData.commonly_found_in}</p>
                </div>

                <div className="detail-section">
                  <h4>origin</h4>
                  <p>{ingredientData.natural_or_synthetic}</p>
                </div>

                <div className="detail-section">
                  <h4>safety status</h4>
                  <p>{ingredientData.safety_status}</p>
                </div>
                
                <div className="detail-section">
                  <h4>health concerns</h4>
                  <p>{ingredientData.health_concerns}</p>
                </div>

                <div className="detail-section">
                  <h4>recommended intake</h4>
                  <p>{ingredientData.recommended_intake}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Login/Register Modal */}
      <LoginRegisterModal 
        isOpen={showLoginModal}
        onClose={() => setShowLoginModal(false)}
        onLoginSuccess={(token, email) => {
          localStorage.setItem('token', token);
          localStorage.setItem('username', email);
          setShowLoginModal(false);
          // Force a re-render to show the save options
          setIsLoggedIn(true);
        }}
      />
    </div>
  );
};

export default HealthResults; 