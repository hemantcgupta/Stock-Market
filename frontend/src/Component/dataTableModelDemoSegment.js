import React, { Component } from 'react';
import Modal from 'react-bootstrap-modal';
import DataTableComponent from '../Component/DataTableComponent'
import { makeStyles } from '@material-ui/core/styles';
import "./component.scss";

let CustomerSegmentationDetails = [];

class DatatableModelDemoSegment extends Component {
    constructor(props) {
        super(props);
        this.state = {
            segmentData: [],
            loading: true,
            tableModalVisible: false,
            segmentRowClassRule: {
                'highlight-dBlue-row': 'data.highlightMe',
            },
        }
    }

    componentWillReceiveProps(newProps) {
        if (newProps.rowData !== this.props.rowData) {
            CustomerSegmentationDetails = newProps.rowData
            this.setState({ segmentData: this.getSegmentsAllData(newProps.rowData), tableModalVisible: newProps.tableModalVisible })
        }
    }

    getSegmentsAllData(segmentData) {
        let custSegment2 = ''
        let data = segmentData.filter((data, index) => {
            let val = data.Seg;
            if (val.indexOf('Total of ')>-1) {
                if(index === 0){
                    data.Seg = '▼ ' + data.Seg.replace("Total of ","");
                    custSegment2 = val.replace("Total of ","");
                }else{
                    data.Seg = '► ' + data.Seg.replace("Total of ","");
                }
            }
            if (val.indexOf('Total of ')>-1) {
                data.highlightMe = true;
            }
            if (data.Segmentation == custSegment2 || val.indexOf('Total of ')>-1){
                return data
            }
        })
        this.setState({ loading: false })
        return data;
    }

    getHighlightedRow(updatedSegmentData, Total_Name) {
        let data = updatedSegmentData.map((data, index) => {
            var total = data.Seg;
            if (Total_Name === total) {
                data.highlightMe = true;
            }
            return data;
        })
        return data;
    }

    totalRowClick(params) {
        this.setState({ loading: true })
        var data = params.data.Seg;
        var selectedRow = data.replace("► ","");
        selectedRow = selectedRow.replace("▼ ","");

        if (data.indexOf('► ')>-1 || data.indexOf('▼ ')>-1) {
            const updatedSegmentData = CustomerSegmentationDetails.filter((d) => {
                let val = d.Seg;
                console.log(val,'value')
                if (val.indexOf('► ')>-1 || val.indexOf('▼ ')>-1) {
                    d.Seg = d.Seg.replace("▼ ","");
                    d.Seg = d.Seg.replace("► ","");
                    if (val.indexOf(data)>-1) {
                        d.Seg = '▼ ' + d.Seg;
                    }else{
                        d.Seg = '► ' + d.Seg;
                    }
                }
                if (d.Segmentation === selectedRow || val.indexOf('► ')>-1 || val.indexOf('▼ ')>-1) {
                    return d;
                }
            })
            this.setState({ segmentData: this.getHighlightedRow(updatedSegmentData, data) })
        }
        this.setState({ loading: false })
    }

    closeModal() {
        this.setState({ tableModalVisible: false })
    }

    render() {
        const { datas, rowData, columns } = this.props;
        return (
            <div>
                <Modal
                    show={this.state.tableModalVisible}
                    onHide={() => this.closeModal()}
                    aria-labelledby="ModalHeader"
                >
                    <Modal.Header closeButton>
                        <Modal.Title id='ModalHeader'>{this.props.header}</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <div className={'root'}>
                            <DataTableComponent
                                rowData={ this.state.segmentData }
                                columnDefs={columns}
                                autoHeight={'autoHeight'}
                                loading={this.state.loading}
                                onCellClicked={(cellData) => this.totalRowClick(cellData)}
                                rowClassRules={this.state.segmentRowClassRule}
                            />
                        </div>
                    </Modal.Body>
                </Modal>
            </div>

        );
    }
}

export default DatatableModelDemoSegment;




