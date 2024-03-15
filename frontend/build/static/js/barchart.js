function top5_O_Ds_barChart(data){

  var margin = {top: 10, right: 20, bottom: 60, left: 70},
  width = 900 - margin.left - margin.right,
  height = 350 - margin.top - margin.bottom;

  // set the ranges
  var x = d3.scale.ordinal().rangeRoundBands([0, width], .05);

  var y = d3.scale.linear().range([height, 0]);

  // define the axis
  var xAxis = d3.svg.axis()
  .scale(x)
  .orient("bottom")


  var yAxis = d3.svg.axis()
  .scale(y)
  .orient("left")
  .ticks(6)
  .tickFormat(d3.format("s"))
  .innerTickSize(-width)
  .outerTickSize(0)
  .tickPadding(10);

  var tip = d3.tip()
  .attr('class', 'd3-tip')
  .offset([-10, 0])
  .html(function(d) {
    return "<strong>CYRevenue:</strong> <span style='color:red'>" + d.CYRevenue + "</span>";
  })

  // add the SVG element
  var svg = d3.select("#top5o_ds_barChart").append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", 
    "translate(" + margin.left + "," + margin.top + ")")
  // .attr('fill', '#d9d9d9');

  svg.call(tip);

  // load the data
  // d3.json("orignal_data.json", function(error, data) {

    data.forEach(function(d) {
      d.CommonRoute = d.CommonRoute;
      d.CYRevenue = +d.CYRevenue;
      console.log(d.CYRevenue)
    });

  // scale the range of the data
  x.domain(data.map(function(d) { return d.CommonRoute; }));
  y.domain([0, d3.max(data, function(d) { return d.CYRevenue; })]);

  // add axis

  var xheight = height+1

  svg.append("g")
  .attr("class", "x bar_axis")
  .attr("transform", "translate(0," + xheight + ")")
  .call(xAxis)
  .selectAll("text")
  .style("text-anchor", "end")
  .attr("dx", "-.8em")
  .attr("dy", "-.55em")
  .attr("transform", "translate(10,5)rotate(-45)" );

  svg.append("g")
  .attr("class", "y bar_axis")
  .call(yAxis)
  .append("text")
  .attr("transform", "rotate(-90)")
  .attr("y",-95)
  .attr("x", "-6em")
  .attr("dy", "4em")
  .style("text-anchor", "end")
  .text("CYRevenueuency");
  // Add bar chart
  svg.selectAll("bar")
  .data(data)
  .enter().append("rect")
  .attr("class", "dash_barChart")
  .attr("x", function(d) { return x(d.CommonRoute); })
  .attr("width", x.rangeBand())
  .attr("height", function(d) { return  y(d.CYRevenue); })
  .attr("y", function(d) {  return height - y(d.CYRevenue); })
  .on('mouseover', tip.show)
  .on('mouseout', tip.hide)
  .transition()
  .attr('y', function(d){
    return y(d.CYRevenue);
  })
  .attr('height', function(d){
    return height - y(d.CYRevenue);
  })
  .delay(function(d,i){
    return i*20;
  })
  .duration(5000)
  // });

}
