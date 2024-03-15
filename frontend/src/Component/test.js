// import React, { Component } from 'react';
// import { render } from 'react-dom';
// import { AgGridReact } from 'ag-grid-react';
// import 'ag-grid-enterprise';
// // import { AgGridReact } from '@ag-grid-community/react';
// // import { AllModules } from '@ag-grid-enterprise/all-modules';
// import 'ag-grid-community/dist/styles/ag-grid.css';
// import 'ag-grid-community/dist/styles/ag-theme-balham.css';
// import '../ag-grid-table.scss';

// class Test extends Component {
//   constructor(props) {
//     super(props);

//     this.state = {
//       columnDefs: [
//         {
//           headerName: 'Hiii',
//           field: 'Cabin',
//           rowGroup: true,
//           editable: true,
//           hide: true,
//         },
//         {
//           headerName: 'RBD',
//           field: 'RBD',
//         },
//         {
//           headerName: 'Tyu',
//           field: 'a',
//           type: 'valueColumn',
//         },
//         {
//           headerName: 'Ryu',
//           field: 'b',
//           type: 'valueColumn',
//         },
//         {
//           headerName: 'Cyu',
//           field: 'c',
//           type: 'valueColumn',
//         },
//         {
//           headerName: 'dyi',
//           field: 'd',
//           type: 'valueColumn',
//         }
//       ],
//       // defaultColDef: {
//       //   flex: 1,
//       //   sortable: true,
//       // },
//       autoGroupColumnDef: { minWidth: 100 },
//       columnTypes: {
//         valueColumn: {
//           // editable: true,
//           aggFunc: 'sum',
//           valueParser: 'Number(newValue)',
//           cellClass: 'number-cell',
//           cellRenderer: 'agAnimateShowChangeCellRenderer',
//           filter: 'agNumberColumnFilter',
//         },
//         totalColumn: {
//           cellRenderer: 'agAnimateShowChangeCellRenderer',
//           // cellClass: 'number-cell',
//         },
//       },
//       rowData: getRowData(),
//       groupDefaultExpanded: 'F',
//     };
//   }

//   onGridReady = params => {
//     this.gridApi = params.api;
//     this.gridColumnApi = params.columnApi;
//   };

//   render() {
//     return (
//       <div style={{ width: '100%', height: '100vh' }}>
//         <div
//           id="myGrid"
//           style={{
//             height: '100%',
//             width: '100%',
//           }}
//           className="ag-theme-balham-dark ag-grid-table"
//         >
//           <AgGridReact
//             // modules={this.state.modules}
//             columnDefs={this.state.columnDefs}
//             defaultColDef={this.state.defaultColDef}
//             autoGroupColumnDef={this.state.autoGroupColumnDef}
//             columnTypes={this.state.columnTypes}
//             rowData={this.state.rowData}
//             groupDefaultExpanded={this.state.groupDefaultExpanded}
//             suppressAggFuncInHeader={true}
//             animateRows={true}
//             onGridReady={this.onGridReady}
//           />
//         </div>
//       </div>
//     );
//   }
// }

// function getRowData() {
//   var rowData = [];
//   rowData.push({
//     'Cabin': 'F',
//     'RBD':'U',
//     'a': 30,
//     'b': 20,
//     'c': 50,
//     'd': 10,
//   }, {
//     'Cabin': 'F',
//     'RBD':'U',
//     'a': 30,
//     'b': 20,
//     'c': 50,
//     'd': 10,
//   }, {
//     'Cabin': 'J',
//     'RBD':'U',
//     'a': 30,
//     'b': 20,
//     'c': 50,
//     'd': 10,
//   }, {
//     'Cabin': 'J',
//     'RBD':'U',
//     'a': 30,
//     'b': 20,
//     'c': 50,
//     'd': 10,
//   })
//   return rowData;
// }

// export default Test;