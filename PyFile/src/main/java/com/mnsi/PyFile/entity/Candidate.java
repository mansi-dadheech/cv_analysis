package com.mnsi.PyFile.entity;

public class Candidate{
	private String name;
	private String cgpa;
	private String isPlaced;
	private String skills;
	public Candidate() {
	
	}
	public Candidate(String name, String cgpa, int isPlaced, String skills) {
		this.name = name;
		this.cgpa = cgpa;
		this.isPlaced = isPlaced == 1 ? "YES":"NO";
		this.skills = skills;
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public String getCgpa() {
		return cgpa;
	}
	public void setCgpa(String cgpa) {
		this.cgpa = cgpa;
	}
	public String getIsPlaced() {
		return isPlaced;
	}
	public void setIsPlaced(int isPlaced) {
		this.isPlaced = isPlaced == 1 ? "YES" : "NO";
	}
	public String getSkills() {
		return skills;
	}
	public void setSkills(String skills) {
		this.skills = skills;
	}
	@Override
	public String toString() {
		return "Candidate [name=" + name + ", cgpa=" + cgpa + ", isPlaced=" + isPlaced + ", skills=" + skills + "]";
	}
	
	
}