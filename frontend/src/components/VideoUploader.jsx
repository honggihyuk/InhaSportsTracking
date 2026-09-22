export function VideoUploader({ onVideoUpload }) {
  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file && onVideoUpload) {
      try {
        await onVideoUpload(file);
      } catch (error) {
        console.error('비디오 업로드 오류:', error);
        alert(`업로드 실패: ${error.message}`);
      }
    }
  };

  return (
    <div className="video-uploader">
      <label className="upload-label">
        📁 비디오 파일 업로드
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="file-input"
        />
      </label>
      <p className="upload-hint">MP4, AVI, MOV 형식 지원</p>
    </div>
  );
}
