import React, { Component } from 'react';
import Modal from 'react-bootstrap-modal';
import MUIDataTable from 'mui-datatables';
import { MuiThemeProvider } from '@material-ui/core/styles';
import theme from '../postableStyle.js';
import MultiLineChart from './MultiLineChart';
import "./component.scss";


class ChartModelDetails extends Component {
    render() {
        return (
            <div>

                <Modal
                    show={this.props.chartVisible}
                    onHide={this.props.closeChartModal}
                    aria-labelledby="ModalHeader"
                >
                    <Modal.Header closeButton>
                        <Modal.Title id='ModalHeader'>{this.props.displayName}</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <MultiLineChart
                            route={this.props.route}
                            displayName={this.props.displayName}
                            forecast={this.props.forecast}
                            gettingYear={this.props.gettingYear}
                            gettingMonth={this.props.gettingMonth}
                            alert={this.props.alert}
                            selectedData={this.props.selectedData}
                            isDirectPOS={this.props.isDirectPOS}
                        />
                    </Modal.Body>
                </Modal>
            </div>

        );
    }
}
export default ChartModelDetails;




