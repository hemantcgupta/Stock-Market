import React, { Component } from 'react';
import Modal from 'react-bootstrap-modal';
import DataTableComponent from '../Component/DataTableComponent';
// import ExpansionPanel from '@material-ui/core/ExpansionPanel';
// import ExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails';
// import ExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary';
// import Typography from '@material-ui/core/Typography';
// import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import TotalRow from '../Component/TotalRow';
import "./component.scss";

let RBDDetails = [];

class DatatableModelDetails extends Component {
    constructor(props) {
        super(props);
        this.state = {
            rbdData: [],
            totalFClick: false,
            totalYClick: true,
            totalJClick: false,
            tableModalVisible: false,
            rbdRowClassRule: {
                'highlight-dBlue-row': 'data.highlightMe',
            },
        }
    }

    componentWillReceiveProps(newProps) {
        if (newProps.rowData !== this.props.rowData) {
            RBDDetails = newProps.rowData
            this.setState({ rbdData: this.getYCabinData(newProps.rowData), tableModalVisible: newProps.tableModalVisible })
        }
    }

    getYCabinData(rbdData) {
        let data = rbdData.filter((data, index) => {
            if (data.RBD === 'Total of F') {
                data.RBD = '► Total of F'
            }
            if (data.RBD === 'Total of Y') {
                data.RBD = '▼ Total of Y'
            }
            if (data.RBD === 'Total of J') {
                data.RBD = '► Total of J'
            }
            if (data.RBD === '► Total of F' || data.RBD === '▼ Total of Y' || data.RBD === '► Total of J') {
                data.highlightMe = true;
            }
            if (data.Cabin === 'Y' || data.RBD === `▼ Total of Y` || data.RBD === `► Total of F` || data.RBD === `► Total of J`) {
                return data;
            }
        })
        return data;
    }

    getHighlightedRow(rbdaData, Total_Name) {
        let data = rbdaData.map((data, index) => {
            var total = data.RBD;
            if (Total_Name === total) {
                data.highlightMe = true;
            }
            return data;
        })
        return data;
    }

    totalRowClick(params) {
        var column = params.colDef.field;
        var data = params.data.RBD;

        if (data === '► Total of Y') {

            const updatedRBDData = RBDDetails.filter((d) => {
                if (d.RBD === '► Total of Y') {
                    d.RBD = '▼ Total of Y'
                }
                if (d.RBD === '▼ Total of F' || d.RBD === '► Total of F') {
                    d.RBD = '► Total of F'
                }
                if (d.RBD === '► Total of J' || d.RBD === '▼ Total of J') {
                    d.RBD = '► Total of J'
                }
                if (d.Cabin === 'Y' || d.RBD === `▼ Total of Y` || d.RBD === `► Total of F` || d.RBD === `► Total of J`) {
                    return d;
                }
            })
            this.setState({ rbdData: this.getHighlightedRow(updatedRBDData, data) })

        } else if (data === '► Total of J') {

            const updatedRBDData = RBDDetails.filter((d) => {
                if (d.RBD === '► Total of J') {
                    d.RBD = '▼ Total of J'
                }
                if (d.RBD === '▼ Total of F' || d.RBD === '► Total of F') {
                    d.RBD = '► Total of F'
                }
                if (d.RBD === '► Total of Y' || d.RBD === '▼ Total of Y') {
                    d.RBD = '► Total of Y'
                }
                if (d.Cabin === 'J' || d.RBD === `► Total of Y` || d.RBD === `► Total of F` || d.RBD === `▼ Total of J`) {
                    return d;
                }
            })
            this.setState({ rbdData: this.getHighlightedRow(updatedRBDData, data) })

        } else if (data === '► Total of F') {

            const updatedRBDData = RBDDetails.filter((d) => {
                if (d.RBD === '► Total of F') {
                    d.RBD = '▼ Total of F'
                }
                if (d.RBD === '▼ Total of Y' || d.RBD === '► Total of Y') {
                    d.RBD = '► Total of Y'
                }
                if (d.RBD === '► Total of J' || d.RBD === '▼ Total of J') {
                    d.RBD = '► Total of J'
                }
                if (d.Cabin === 'F' || d.RBD === `► Total of Y` || d.RBD === `▼ Total of F` || d.RBD === `► Total of J`) {
                    return d;
                }
            })
            this.setState({ rbdData: this.getHighlightedRow(updatedRBDData, data) })
        }

    }

    closeModal() {
        this.setState({ tableModalVisible: false })
    }


    render() {
        const { datas, rowData, interline, columns, totalData } = this.props;
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
                        <div className={'root'} style={{maxHeight: "80vh", overflowY: "scroll"}}>
                            {/* {datas.map((data, i) => {
                                return (
                                    <ExpansionPanel expanded={this.state.expanded === data[0].Cabin} onChange={this.handleChange(data[0].Cabin)}>
                                        <ExpansionPanelSummary
                                            expandIcon={<ExpandMoreIcon />}
                                            aria-controls="panel1bh-content"
                                            id="panel1bh-header"
                                        >
                                            <Typography className={'heading'}>{data[0].Cabin}</Typography>
                                        </ExpansionPanelSummary>
                                        <ExpansionPanelDetails> */}
                            <DataTableComponent
                                rowData={interline ? rowData : this.state.rbdData}
                                columnDefs={columns}
                                autoHeight={'autoHeight'}
                                onCellClicked={(cellData) => this.totalRowClick(cellData)}
                                rowClassRules={this.state.rbdRowClassRule}
                                getRowStyle={this.props.getRowStyle}
                            />
                            {totalData && (<TotalRow
                                rowData={totalData}
                                columnDefs={columns}
                                responsive={true}
                                reducingPadding={true}
                            />)
                            }

                            {/* </ExpansionPanelDetails>
                                    </ExpansionPanel>
                                )
                            })} */}
                        </div>
                    </Modal.Body>
                </Modal>
            </div>

        );
    }
}

export default DatatableModelDetails;




