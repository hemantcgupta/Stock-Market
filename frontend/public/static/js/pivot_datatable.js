function channelPivot(channalPivotTable){

	$.each( channalPivotTable, function( key, value ) {
		var num = numberFormat(value.channel,2)
		var num2 = numberFormat(value.cy_revenue,2)
		var num3 = numberFormat(value.ly_revenue,2)
		var vly_revenue = value.vly_revenue.toFixed(2)
		htm = '<tr>';
		htm += '<td>'+ value.channel + '</td>';
		htm += '<td>'+ num2 + '</td>';
		htm += '<td>'+ num3 + '</td>';
		htm += '<td>'+ vly_revenue + '</td>';
		htm += '</tr>';

        $("#channel_pivot_tabledata").append(htm);
    });

	datatablePivot("datatable-pivot");
}

function top10Routes(top10RoutePivotTable){

	$.each( top10RoutePivotTable, function( key, value ) {

		htm = '<tr>';
		htm += '<td>'+ value.commonroute + '</td>';
		htm += '<td>'+ value.cy_revenue + '</td>';
		htm += '<td>'+ value.ly_revenue + '</td>';
		htm += '<td>'+ value.vly_revenue + '</td>';
		htm += '</tr>';

        $("#top10Route_tabledata").append(htm);
    });

	datatablePivot("datatable-top10Routes");
}

function datatablePivot(datatableId){
	if ($("#"+datatableId).length) {
	$("#"+datatableId).DataTable({
	  fixedHeader: true,
	  dom: "Blfrtip",
	  buttons: [
		{
		  extend: "copy",
		  className: "btn-sm"
		},
		{
		  extend: "csv",
		  className: "btn-sm"
		},
		{
		  extend: "excel",
		  className: "btn-sm"
		},
		{
		  extend: "pdfHtml5",
		  className: "btn-sm"
		},
		{
		  extend: "print",
		  className: "btn-sm"
		},
	  ],
	  // responsive: false
	});
};
}