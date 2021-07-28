package com.mnsi.PyFile.controller;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import com.mnsi.PyFile.entity.Candidate;


@Controller
public class PyController {
	
	@GetMapping("/")
	public String runPyFile() {
		return "cv-home";
	}
	
	@PostMapping("/save")
	public String saveData(@RequestParam("file") MultipartFile file,Model model) {
		String uploadPath = "";
		try{
            byte[] bytes = file.getBytes();
            String upFile;
            upFile = FileSystems.getDefault().getPath("DOCS").toAbsolutePath()+"/"; //Paths.get("./DOCS" + FileSystems.getDefault().getSeparator()).normalize().toAbsolutePath().toString();
            System.out.println("upload PATH: " + upFile);
            Path path = Paths.get(upFile + file.getOriginalFilename());
//            Path path = Paths.get("E://KMS" + file.getOriginalFilename());
//            Path path = Paths.get(indexConstants.uploadedDocumentPath + file.getOriginalFilename());
            uploadPath = FileSystems.getDefault().getPath("DOCS").toAbsolutePath()+"\\" + file.getOriginalFilename();
            
            System.out.println("upload Path is: " + uploadPath);
            Files.write(path, bytes);
            System.out.println(file.getOriginalFilename() + " file successfully uploaded");
        }
        catch (IOException e){
            System.out.println("ERROR : " + e.getMessage());
        }
	    try {
	    	String param = uploadPath;
			Process p = Runtime.getRuntime().exec("python E://majorProject//testing_model.py " + uploadPath);
	    	BufferedReader bf =  new BufferedReader(new InputStreamReader(p.getInputStream()));
	    	StringBuilder sb = new StringBuilder();
	    	String line;
	    	while((line = bf.readLine()) != null) {
	    		System.out.println(line);
	    		sb.append(line).append("\n");
	    	}
		System.out.println("value is" + sb.toString());

	    	model.addAttribute("res",sb);	    	
		} catch (IOException e) {			
			e.printStackTrace();
		}
		return "result";
	}
	
	@GetMapping("/loginController")
	public String loginController() {
		return "login-form";
	}
	
	@GetMapping("/admin")
	public String showAdmin(Model model) {
		List<Candidate> candidates = new ArrayList<>();
		String file = "./cdata.csv";
		Path path  = Paths.get(file);
		List<String> dataList = new ArrayList<>();
		try {
			dataList = Files.readAllLines(path);
			dataList.remove(0);
			for(String s: dataList)
				candidates.add(getCandidate(s));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		model.addAttribute("result",candidates);
		return "admin-home";
	}
	@GetMapping("/showDetails")
	public String getDetails() {
		return "show-details";
	}
	public Candidate getCandidate(String s) {
		String[] arr = s.split(",");
		String name = arr[0];
		String cgpa = arr[1];
		int isPlaced = Integer.parseInt(arr[2]);
		StringBuilder skills = new StringBuilder("");
		for(int i = 5;i<arr.length-3;++i) {
			skills.append(arr[i]);			
		}
		Candidate candidate = new Candidate(name,cgpa,isPlaced,skills.toString());
		System.out.println(candidate);
		return candidate;
	}
}