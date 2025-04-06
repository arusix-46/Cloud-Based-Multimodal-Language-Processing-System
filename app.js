import React, { useState } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("es");
  const [translatedText, setTranslatedText] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  const [transcription, setTranscription] = useState("");

  // 🟢 Function to Translate Text
  const handleTranslate = async () => {
    const response = await axios.post("http://localhost:5000/translate", {
      text,
      target_language: language,
    });
    setTranslatedText(response.data.translated_text);
  };

  // 🟢 Function to Convert Text to Speech
  const handleTextToSpeech = async () => {
    const response = await axios.post("http://localhost:5000/text-to-speech", {
      text,
      language_code: language,  // Language selection
    }, { responseType: "blob" }); // Get binary data (MP3)
    
    const audioBlob = new Blob([response.data], { type: "audio/mp3" });
    const audioURL = URL.createObjectURL(audioBlob);
    setAudioUrl(audioURL);
  };

  // 🟢 Function to Convert Speech to Text
  const handleSpeechToText = async () => {
    const formData = new FormData();
    formData.append("file", audioFile);

    const response = await axios.post("http://localhost:5000/speech-to-text", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    setTranscription(response.data.transcription);
  };

  return (
    <div style={{ textAlign: "center", fontFamily: "Arial, sans-serif" }}>
      <h2>🌍 Cloud-Based Language Processing</h2>

      {/* 🟢 Text Input for Translation */}
      <textarea 
        value={text} 
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text here..."
        rows="4"
        cols="50"
      />
      <br />

      {/* 🟢 Language Selection */}
      <select value={language} onChange={(e) => setLanguage(e.target.value)}>
        <option value="es">Spanish</option>
        <option value="fr">French</option>
        <option value="de">German</option>
        <option value="hi">Hindi</option>
        <option value="zh">Chinese</option>
      </select>
      <br /><br />

      {/* 🟢 Buttons for Actions */}
      <button onClick={handleTranslate}>Translate</button>
      <button onClick={handleTextToSpeech}>Text-to-Speech</button>
      <br /><br />

      {/* 🟢 Display Translated Text */}
      <h3>📝 Translated Text: {translatedText}</h3>

      {/* 🟢 Audio Player for Text-to-Speech */}
      {audioUrl && (
        <div>
          <h3>🔊 Generated Audio:</h3>
          <audio controls>
            <source src={audioUrl} type="audio/mp3" />
          </audio>
        </div>
      )}

      {/* 🟢 Upload Audio File for Speech-to-Text */}
      <h3>🎤 Upload Audio for Speech-to-Text:</h3>
      <input type="file" accept="audio/*" onChange={(e) => setAudioFile(e.target.files[0])} />
      <button onClick={handleSpeechToText}>Convert Speech to Text</button>

      {/* 🟢 Display Transcribed Text */}
      <h3>📜 Transcribed Text: {transcription}</h3>
    </div>
  );
}

export default App;
