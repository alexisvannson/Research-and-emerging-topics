document.getElementById("generateBtn").addEventListener("click", async () => {
    const cvFile = document.getElementById("cvFile").files[0];
    const jobDescription = document.getElementById("jobDescription").value;
    const status = document.getElementById("status");
  
    if (!cvFile || !jobDescription) {
      status.textContent = "Please upload a CV and enter a job description.";
      return;
    }
  
    const formData = new FormData();
    formData.append("cv", cvFile);
    formData.append("jobDescription", jobDescription);
  
    try {
      status.textContent = "Generating cover letter...";
      const response = await fetch("YOUR_BACKEND_ENDPOINT", {
        method: "POST",
        body: formData
      });
  
      if (!response.ok) throw new Error("Failed to generate cover letter.");
  
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "cover_letter.pdf";
      link.click();
      URL.revokeObjectURL(url);
  
      status.textContent = "Cover letter generated successfully!";
    } catch (error) {
      status.textContent = `Error: ${error.message}`;
    }
  });  