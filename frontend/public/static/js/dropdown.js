function dropdownFilter(obj, array,filter){
				console.log(obj, array,filter , 'dropdownfilter') 
				var heirarchy=["RegionName","CountryName","POS"]
				var get_obj = "d3.nest()"
				for(i=0; i < array.length+filter ; i++){
				var get_obj = get_obj + " .key(function(d,i) { return d['"+heirarchy[i]+"'] }) " } 
				var get_obj = get_obj + ".entries(obj)"
				//console.log(get_obj)			
				var get_obj = eval(get_obj)	
				array.map(function(e,i){
				var Key = get_obj.map(function (f,j){ return f.key  })
				get_obj= get_obj[Key.indexOf(e)].values				
				})				
				return get_obj
				}

function dropdownCreate(id,datas){
	console.log(datas,'datassssssssssss')
	datas.unshift("----Select----");
	deleteElement(id);
	d3.select(id)
          .selectAll("option")
          .data(datas)
          .enter()
          .append("option")
          .attr("value", function(e) { return e })         
          .text(function(f) { return f; });

}


function deleteElement(id){
	d3.select(id)
          .selectAll("option").remove();
}


//Data population start here

function Dropdown_gen(data) {

	console.log(data,'dropdown_gen')
	var arrayHeirarchy = new Array();
	var filterData =  new Object();


	var group = d3.nest()
		.key(function(d, i) { return d.RegionName; }) 
		.entries(data);	
		// console.log(data,'group')	
		var keyy = group.map(function(d,i){
			 // console.log(d.key,'keeyy')
	 		return d.key	
		})


dropdownCreate("#region",keyy)


$("#region").change(function(d,i){	
	arrayHeirarchy=[]	
	arrayHeirarchy[0] = d.target.value			
	var filterData= dropdownFilter(data,arrayHeirarchy,1)	

	var keyy1 = filterData.map(function(d,i){
		return d.key	
	})			
	dropdownCreate("#country",keyy1)
	})

$("#country").change(function(d,i){	
	arrayHeirarchy[1] = d.target.value			
	var filterData= dropdownFilter(data, arrayHeirarchy,1)			
	var keyy2 = filterData.map(function(d,i){
		return d.key	
	})			
	dropdownCreate("#city",keyy2)

})

}
